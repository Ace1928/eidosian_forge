import warnings
import numpy as np
import torch
import tree
from packaging.version import parse
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def _should_torch_compile(self):
    if self.jit_compile and parse(torch.__version__) < parse('2.1.0'):
        warnings.warn('Please upgrade to torch>=2.1.0 for `jit_compile=True` to take effect. Using `jit_compile=False`')
        self.jit_compile = False
    return self.jit_compile