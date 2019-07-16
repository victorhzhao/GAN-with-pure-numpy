# GAN in Numpy
This is a very simple step by step implementation of GAN using only numpy.  
Without the use of GPU, it may takes too much time to generate all the numbers.  
To get the result quickly using only CPU, I suggest working with one number.  

## How to use
In bash or command line(windows) or powershell under this directory

```bash
$ python gan.py
#this will generator a random number [0, 9]

$ python gan.py 9 0 8
#add the number(s) you want the program to generate, e.g. 0(to generate 0) or 0 8(to generate 0 and 8)
```

### What's included 
* Vanilla GAN
* Xavier Initialization
* SGD

### Requirements  
* Numpy  
* Matplotlib/PIL (to visualize/save results)  

## Network  
![network](./results/network.png)

## Results
![7](./results/7.png)

## image per epoch
![epoch 0](./results/epoch_000.png)
![epoch 1](./results/epoch_001.png)
![epoch 2](./results/epoch_002.png)
![epoch 3](./results/epoch_003.png)
![epoch 4](./results/epoch_004.png)
![epoch 5](./results/epoch_005.png)
![epoch 6](./results/epoch_006.png)
![epoch 7](./results/epoch_007.png)
![epoch 8](./results/epoch_008.png)
![epoch 9](./results/epoch_009.png)
![epoch 10](./results/epoch_010.png)
![epoch 11](./results/epoch_011.png)
![epoch 12](./results/epoch_012.png)
![epoch 13](./results/epoch_013.png)
![epoch 14](./results/epoch_014.png)
![epoch 15](./results/epoch_015.png)
![epoch 16](./results/epoch_016.png)
![epoch 17](./results/epoch_017.png)
![epoch 18](./results/epoch_018.png)
![epoch 19](./results/epoch_019.png)


#### reference
[generative adversarial networks](https://arxiv.org/pdf/1406.2661.pdf)
