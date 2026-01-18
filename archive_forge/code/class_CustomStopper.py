import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from ray.tune.examples.mnist_pytorch import train_func, test_func, ConvNet,\
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
class CustomStopper(tune.Stopper):

    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 5 if args.smoke_test else 100
        if not self.should_stop and result['mean_accuracy'] > 0.96:
            self.should_stop = True
        return self.should_stop or result['training_iteration'] >= max_iter

    def stop_all(self):
        return self.should_stop