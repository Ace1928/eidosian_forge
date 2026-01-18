import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _create_device_transfers(self, tensors):
    """Encode inter-device transfers if the current device
    is not the same as the Staging Area's device.
    """
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    curr_device_scope = control_flow_ops.no_op().device
    if curr_device_scope != self._coloc_op.device:
        tensors = [array_ops.identity(t) for t in tensors]
    return tensors