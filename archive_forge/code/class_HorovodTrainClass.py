import argparse
import os
import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.horovod import HorovodTrainer
class HorovodTrainClass:

    def __init__(self, config):
        self.log_interval = config.get('log_interval', 10)
        self.use_cuda = config.get('use_cuda', False)
        if self.use_cuda:
            torch.cuda.set_device(hvd.local_rank())
        self.model, self.optimizer, self.train_loader, self.train_sampler = setup(config)

    def train(self, epoch):
        loss = train_epoch(self.model, self.optimizer, self.train_sampler, self.train_loader, epoch, self.log_interval, self.use_cuda)
        return loss