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
def _purge_model_variables(self, trainable_variables=False, non_trainable_variables=False, optimizer_variables=False, metrics_variables=False):
    """Remove all the model variable for memory saving.

        During JAX training, since the training function are stateless, we have
        to pass in and get the model weights over and over, during which the
        copy of the weights that attached to the KerasVariable are still and
        occupying extra memory. We remove those variable to save memory (for
        better memory utilization) at the beginning of the epoch, and reattach
        the value back to variables at the end of the epoch, via
        `jax_state_sync()`.
        """
    if trainable_variables:
        for v in self.trainable_variables:
            v._value = None
    if non_trainable_variables:
        for v in self.non_trainable_variables:
            v._value = None
    if optimizer_variables:
        for v in self.optimizer.variables:
            v._value = None
    if metrics_variables:
        for v in self.metrics_variables:
            v._value = None