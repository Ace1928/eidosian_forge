import multiprocessing.dummy
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def _warn_if_super_not_called(self):
    warn = False
    if not hasattr(self, '_workers'):
        self._workers = 1
        warn = True
    if not hasattr(self, '_use_multiprocessing'):
        self._use_multiprocessing = False
        warn = True
    if not hasattr(self, '_max_queue_size'):
        self._max_queue_size = 10
        warn = True
    if warn:
        warnings.warn('Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.', stacklevel=2)