import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
class LearningRateBatchScheduler(keras.callbacks.Callback):
    """Scheduler that dynamically sets the learning rate of model."""

    def __init__(self, update_freq=None):
        self._update_freq = update_freq

    def on_batch_begin(self, batch, logs=None):
        if self._update_freq and batch % self._update_freq != 0:
            return
        lr = 0.001 * (batch % 10)
        keras.backend.set_value(self.model.optimizer.lr, lr)