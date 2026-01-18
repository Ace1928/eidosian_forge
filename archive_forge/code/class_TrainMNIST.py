from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import (
class TrainMNIST(tune.Trainable):

    def setup(self, config):
        use_cuda = config.get('use_gpu') and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.get('lr', 0.01), momentum=config.get('momentum', 0.9))

    def step(self):
        train_func(self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test_func(self.model, self.test_loader, self.device)
        return {'mean_accuracy': acc}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        self.model.load_state_dict(torch.load(checkpoint_path))