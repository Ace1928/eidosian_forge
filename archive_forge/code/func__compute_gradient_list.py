import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _compute_gradient_list(f, xs, delta):
    """Compute gradients for a list of x values."""
    xs = [ops.convert_to_tensor(x) for x in xs]
    xs_dtypes = [x.dtype for x in xs]
    xs_shapes = [x.shape for x in xs]
    f_temp = _prepare(f, xs_dtypes, xs_shapes)
    y = f_temp(*xs)
    return tuple(zip(*[_compute_gradient(f, y.shape, dtypes.as_dtype(y.dtype), xs, i, delta) for i in range(len(xs))]))