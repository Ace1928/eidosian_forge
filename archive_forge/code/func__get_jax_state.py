import collections
import itertools
from functools import partial
import jax
import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import ops
from keras.src import optimizers as optimizers_module
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def _get_jax_state(self, trainable_variables=False, non_trainable_variables=False, optimizer_variables=False, metrics_variables=False, purge_model_variables=False):
    state = []
    if trainable_variables:
        state.append([v.value for v in self.trainable_variables])
    if non_trainable_variables:
        state.append([v.value for v in self.non_trainable_variables])
    if optimizer_variables:
        state.append([v.value for v in self.optimizer.variables])
    if metrics_variables:
        state.append([v.value for v in self.metrics_variables])
    if purge_model_variables:
        self._purge_model_variables(trainable_variables=trainable_variables, non_trainable_variables=non_trainable_variables, optimizer_variables=optimizer_variables, metrics_variables=metrics_variables)
    return tuple(state)