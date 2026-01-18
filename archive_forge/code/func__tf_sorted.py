from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_ops
def _tf_sorted(iterable, key, reverse):
    """Overload of sorted_ for Tensor iterable."""
    if reverse is py_builtins.UNSPECIFIED:
        direction = 'ASCENDING'
    else:
        direction = 'DESCENDING'
    if key is not py_builtins.UNSPECIFIED:
        mapped = parallel_ops.vectorized_map(key, iterable)
        if mapped.shape.rank is not None and mapped.shape.rank != 1:
            raise ValueError('sort only supports only 1D tensors')
        with ops.control_dependencies([check_ops.assert_rank_v2(mapped, 1, 'sort only supports only 1D tensors')]):
            order = sort_ops.argsort(mapped, direction=direction)
            return array_ops.gather_v2(iterable, order)
    if iterable.shape.rank is not None and iterable.shape.rank != 1:
        raise ValueError('sort only supports only 1D tensors')
    with ops.control_dependencies([check_ops.assert_rank_v2(iterable, 1, 'sort only supports only 1D tensors')]):
        return sort_ops.sort(iterable, direction=direction)