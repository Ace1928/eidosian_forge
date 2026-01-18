import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def _mark_as_return(tensor):
    """Marks `tensor` as the return value for automatic control deps."""
    if not tensor_util.is_tf_type(tensor):
        return tensor
    return_tensor = acd.mark_as_return(tensor)
    if getattr(tensor, '_keras_mask', None) is not None:
        return_tensor._keras_mask = acd.mark_as_return(tensor._keras_mask)
    else:
        return_tensor._keras_mask = None
    if getattr(tensor, '_tfp_distribution', None) is not None:
        return_tensor._tfp_distribution = tensor._tfp_distribution
    return return_tensor