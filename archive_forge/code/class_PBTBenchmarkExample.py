import numpy as np
import argparse
import random
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
class PBTBenchmarkExample(tune.Trainable):
    """Toy PBT problem for benchmarking adaptive learning rate.

    The goal is to optimize this trainable's accuracy. The accuracy increases
    fastest at the optimal lr, which is a function of the current accuracy.

    The optimal lr schedule for this problem is the triangle wave as follows.
    Note that many lr schedules for real models also follow this shape:

     best lr
      ^
      |    /      |   /        |  /          | /            ------------> accuracy

    In this problem, using PBT with a population of 2-4 is sufficient to
    roughly approximate this lr schedule. Higher population sizes will yield
    faster convergence. Training will not converge without PBT.
    """

    def setup(self, config):
        self.lr = config['lr']
        self.accuracy = 0.0

    def step(self):
        midpoint = 100
        q_tolerance = 3
        noise_level = 2
        if self.accuracy < midpoint:
            optimal_lr = 0.01 * self.accuracy / midpoint
        else:
            optimal_lr = 0.01 - 0.01 * (self.accuracy - midpoint) / midpoint
        optimal_lr = min(0.01, max(0.001, optimal_lr))
        q_err = max(self.lr, optimal_lr) / min(self.lr, optimal_lr)
        if q_err < q_tolerance:
            self.accuracy += 1.0 / q_err * random.random()
        elif self.lr > optimal_lr:
            self.accuracy -= (q_err - q_tolerance) * random.random()
        self.accuracy += noise_level * np.random.normal()
        self.accuracy = max(0, self.accuracy)
        return {'mean_accuracy': self.accuracy, 'cur_lr': self.lr, 'optimal_lr': optimal_lr, 'q_err': q_err, 'done': self.accuracy > midpoint * 2}

    def save_checkpoint(self, checkpoint_dir):
        return {'accuracy': self.accuracy, 'lr': self.lr}

    def load_checkpoint(self, checkpoint):
        self.accuracy = checkpoint['accuracy']

    def reset_config(self, new_config):
        self.lr = new_config['lr']
        self.config = new_config
        return True