import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def _tf_tensor_len(s):
    """Overload of len_ for Tensor arguments."""
    if s.shape.ndims and s.shape.dims[0].value is not None:
        return s.shape.dims[0].value
    shape = array_ops.shape(s)
    assert shape.shape, 'shape tensor of zero size? {}'.format(shape)
    if shape.shape[0] == 0:
        raise ValueError('len requires a non-scalar tensor, got one of shape {}'.format(shape))
    if shape.shape.dims[0].value is not None:
        return array_ops.shape(s)[0]
    rank = array_ops.rank(s)

    def raise_zero_rank_error():
        msg = gen_string_ops.string_join(['len requires non-zero rank, got ', gen_string_ops.as_string(rank)])
        with ops.control_dependencies([control_flow_assert.Assert(False, [msg])]):
            return constant_op.constant(0, dtype=dtypes.int32)
    return cond.cond(rank > 0, lambda: array_ops.shape(s)[0], raise_zero_rank_error)