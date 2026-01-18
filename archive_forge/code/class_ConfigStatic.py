import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
class ConfigStatic:

    def __init__(self, config):
        object.__setattr__(self, '__dict__', dict(config))

    def __setattr__(self, name, value):
        raise AttributeError('Error: wandb.run.config_static is a readonly object')

    def __setitem__(self, key, val):
        raise AttributeError('Error: wandb.run.config_static is a readonly object')

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(self.__dict__)