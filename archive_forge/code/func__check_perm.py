import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _check_perm(self, perm):
    """Static check of perm."""
    if perm.shape.ndims is not None and perm.shape.ndims < 1:
        raise ValueError(f'Argument `perm` must have at least 1 dimension. Received: {perm}.')
    if not perm.dtype.is_integer:
        raise TypeError(f'Argument `perm` must be integer dtype. Received: {perm}.')
    static_perm = tensor_util.constant_value(perm)
    if static_perm is not None:
        sorted_perm = np.sort(static_perm, axis=-1)
        if np.any(sorted_perm != np.arange(0, static_perm.shape[-1])):
            raise ValueError(f'Argument `perm` must be a vector of unique integers from 0 to {static_perm.shape[-1] - 1}.')