import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def _reshape_for_efficiency(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False):
    """Maybe reshape a, b, and return an inverse map.  For matmul/solve."""

    def identity(x):
        return x
    still_need_to_transpose = True
    if a.shape.ndims is None or b.shape.ndims is None:
        return (a, b, identity, still_need_to_transpose)
    if a.shape.ndims >= b.shape.ndims:
        return (a, b, identity, still_need_to_transpose)
    b_extra_ndims = b.shape.ndims - a.shape.ndims
    b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
    b_main_sh = array_ops.shape(b)[b_extra_ndims:]
    a_domain_sz_ = a.shape[-2 if adjoint_a or transpose_a else -1]
    b_eq_sz_ = b.shape[-2 if adjoint_b or transpose_b else -1]
    b_extra_sz_ = np.prod(b.shape[:b_extra_ndims].as_list()) if b.shape[:b_extra_ndims].is_fully_defined() else None
    if a_domain_sz_ is not None and b_eq_sz_ is not None and (b_extra_sz_ is not None):
        if b_extra_sz_ < 2 or a_domain_sz_ <= b_eq_sz_:
            return (a, b, identity, still_need_to_transpose)
    if adjoint_a:
        a = array_ops.matrix_transpose(a, conjugate=True)
    elif transpose_a:
        a = array_ops.matrix_transpose(a, conjugate=False)
    if adjoint_b:
        b = array_ops.matrix_transpose(b, conjugate=True)
    elif transpose_a:
        b = array_ops.matrix_transpose(b, conjugate=False)
    still_need_to_transpose = False
    b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
    b_main_sh = array_ops.shape(b)[b_extra_ndims:]
    perm = np.concatenate((np.arange(b_extra_ndims, b.shape.ndims), np.arange(0, b_extra_ndims)), 0)
    b_extra_on_end = array_ops.transpose(b, perm=perm)
    b_squashed_end = array_ops.reshape(b_extra_on_end, array_ops.concat((b_main_sh[:-1], [-1]), 0))

    def reshape_inv(y):
        y_extra_shape = array_ops.concat((array_ops.shape(y)[:-1], [b_main_sh[-1]], b_extra_sh), 0)
        y_extra_on_end = array_ops.reshape(y, y_extra_shape)
        inverse_perm = np.argsort(perm)
        return array_ops.transpose(y_extra_on_end, perm=inverse_perm)
    return (a, b_squashed_end, reshape_inv, still_need_to_transpose)