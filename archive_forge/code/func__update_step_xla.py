import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@tf.function(jit_compile=True)
def _update_step_xla(self, gradient, variable, key):
    """A wrapper of `update_step` to enable XLA acceleration.

        Due to `tf.function` tracing mechanism, for (gradient, variable) pairs
        of the same shape and dtype, the execution graph always invoke the first
        pair it has seen. Thus, we need a `key` argument to make each (gradient,
        variable) pair unique. In additions, XLA cannot understand string input,
        so the key is an integer.

        Args:
          gradient: backpropagated gradient of the given variable.
          variable: variable whose value needs to be updated.
          key (int): a unique key that identifies the variable.

        Returns:
          An `Operation` that applies the specified gradients.
        """
    return self._update_step(gradient, variable)