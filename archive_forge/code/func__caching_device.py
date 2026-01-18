import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _caching_device(rnn_cell):
    """Returns the caching device for the RNN variable.

  This is useful for distributed training, when variable is not located as same
  device as the training worker. By enabling the device cache, this allows
  worker to read the variable once and cache locally, rather than read it every
  time step from remote when it is needed.

  Note that this is assuming the variable that cell needs for each time step is
  having the same value in the forward path, and only gets updated in the
  backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
  cell body relies on any variable that gets updated every time step, then
  caching device will cause it to read the stall value.

  Args:
    rnn_cell: the rnn cell instance.
  """
    if context.executing_eagerly():
        return None
    if not getattr(rnn_cell, '_enable_caching_device', False):
        return None
    if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
        logging.warning('Variable read device caching has been disabled because the RNN is in tf.while_loop loop context, which will cause reading stalled value in forward path. This could slow down the training due to duplicated variable reads. Please consider updating your code to remove tf.while_loop if possible.')
        return None
    if rnn_cell._dtype_policy.compute_dtype != rnn_cell._dtype_policy.variable_dtype:
        logging.warning("Variable read device caching has been disabled since it doesn't work with the mixed precision API. This is likely to cause a slowdown for RNN training due to duplicated read of variable for each timestep, which will be significant in a multi remote worker setting. Please consider disabling mixed precision API if the performance has been affected.")
        return None
    return lambda op: op.device