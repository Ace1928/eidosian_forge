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
class MaybeDistributionScope:
    """Provides a context allowing no distribution strategy."""

    def __init__(self, distribution):
        self._distribution = distribution
        self._scope = None

    def __enter__(self):
        if self._distribution:
            self._scope = self._distribution.scope()
            self._scope.__enter__()

    def __exit__(self, exc_type, value, traceback):
        if self._distribution:
            self._scope.__exit__(exc_type, value, traceback)
            self._scope = None