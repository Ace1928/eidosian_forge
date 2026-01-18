import contextlib
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
def _check_if_mixed_precision_graph_rewrite_is_enabled(policy):
    if mixed_precision_global_state.is_mixed_precision_graph_rewrite_enabled():
        raise ValueError('The global dtype policy cannot be set to "{policy.name}", because the mixed precision graph rewrite has already been enabled.\nAt most, one of the following can be called:\n\n  1. tf.compat.v1.train.enable_mixed_precision_graph_rewrite() (You called this first)\n  2. tf.keras.mixed_precision.experimental.set_global_policy() with a mixed precision policy (You called this second)\n\nYou called both functions, which is an error, because both functions enable you to use mixed precision. If in doubt which function to use, use the second, as it supports Eager execution and is more customizable.'.format(policy=policy))