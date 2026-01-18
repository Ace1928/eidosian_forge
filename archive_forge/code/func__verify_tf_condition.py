import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _verify_tf_condition(cond, tag):
    """Ensures that the condition can be used in a TF control flow."""
    extra_hint = 'to check for None, use `is not None`'
    cond = tensor_conversion.convert_to_tensor_v2(cond)
    if cond.dtype != dtypes.bool:
        raise ValueError('condition of {} expected to be `tf.bool` scalar, got {}; to use as boolean Tensor, use `tf.cast`; {}'.format(tag, cond, extra_hint))
    if cond.shape is None or cond.shape.ndims is None:
        cond = array_ops.reshape(cond, ())
    elif cond.shape.ndims > 0:
        known_dims = [d for d in cond.shape.as_list() if d is not None]
        if np.prod(known_dims) > 1:
            raise ValueError('condition of {} expected to be `tf.bool` scalar, got {}; {}'.format(tag, cond, extra_hint))
        else:
            cond = array_ops.reshape(cond, ())
    return cond