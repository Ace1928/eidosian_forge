import warnings
import numpy
from sacred.dependencies import get_digest
from sacred.observers import RunObserver
import wandb
def __update_config(self, config):
    for k, v in config.items():
        self.run.config[k] = v
    self.run.config['resources'] = []