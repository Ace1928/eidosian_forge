import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _default_inner_shape_for_pylist(pylist, ragged_rank):
    """Computes a default inner shape for the given python list."""

    def get_inner_shape(item):
        """Returns the inner shape for a python list `item`."""
        if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
            return ()
        elif len(item) > 0:
            return (len(item),) + get_inner_shape(item[0])
        return (0,)

    def check_inner_shape(item, shape):
        """Checks that `item` has a consistent shape matching `shape`."""
        is_nested = isinstance(item, (list, tuple)) or np.ndim(item) != 0
        if is_nested != bool(shape):
            raise ValueError('inner values have inconsistent shape')
        if is_nested:
            if shape[0] != len(item):
                raise ValueError('inner values have inconsistent shape')
            for child in item:
                check_inner_shape(child, shape[1:])
    flat_values = pylist
    for dim in range(ragged_rank):
        if not all((isinstance(v, (list, tuple)) or np.ndim(v) != 0 for v in flat_values)):
            raise ValueError('pylist has scalar values depth %d, but ragged_rank=%d requires scalar value depth greater than %d' % (dim + 1, ragged_rank, ragged_rank))
        flat_values = sum((list(v) for v in flat_values), [])
    inner_shape = get_inner_shape(flat_values)
    check_inner_shape(flat_values, inner_shape)
    return inner_shape[1:]