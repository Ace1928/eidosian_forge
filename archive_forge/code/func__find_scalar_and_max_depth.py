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
def _find_scalar_and_max_depth(pylist):
    """Finds nesting depth of scalar values in pylist.

  Args:
    pylist: A nested python `list` or `tuple`.

  Returns:
    A tuple `(scalar_depth, max_depth)`.  `scalar_depth` is the nesting
    depth of scalar values in `pylist`, or `None` if `pylist` contains no
    scalars.  `max_depth` is the maximum depth of `pylist` (including
    empty lists).

  Raises:
    ValueError: If pylist has inconsistent nesting depths for scalars.
  """
    if isinstance(pylist, (list, tuple)) or np.ndim(pylist) != 0:
        scalar_depth = None
        max_depth = 1
        for child in pylist:
            child_scalar_depth, child_max_depth = _find_scalar_and_max_depth(child)
            if child_scalar_depth is not None:
                if scalar_depth is not None and scalar_depth != child_scalar_depth + 1:
                    raise ValueError('all scalar values must have the same nesting depth')
                scalar_depth = child_scalar_depth + 1
            max_depth = max(max_depth, child_max_depth + 1)
        return (scalar_depth, max_depth)
    return (0, 0)