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
def _compute_numeric_jacobian(f, y_size, y_dtype, xs, param, delta):
    """Computes the numeric Jacobian for f regarding xs[param].

  One can think of the relation among f, xs and y as y = f(xs).

  Args:
    f: the function.
    y_size: the number of elements of the result.
    y_dtype: the dtype of the result.
    xs: a list of tensors.
    param: the index of the target parameter.
    delta: the amount of perturbation we give to the input.

  Returns:
    A 2-d numpy array representing the Jacobian. It has "y_size" rows
    and "x_size" columns where "x_size" is the number of elements in xs[param]
    and "y_size" is the number of elements in the result.
  """
    x_shape = xs[param].shape
    x_dtype = xs[param].dtype
    x_size = _product(x_shape) * (2 if x_dtype.is_complex else 1)
    y_size = y_size * (2 if y_dtype.is_complex else 1)
    x_dtype = x_dtype.real_dtype.as_numpy_dtype
    y_dtype = y_dtype.real_dtype.as_numpy_dtype
    xs_dtypes = [x.dtype for x in xs]
    xs_shapes = [x.shape for x in xs]
    xs = [np.asarray(_to_numpy(x)) for x in xs]
    x = xs[param]
    scale = np.asarray(2 * delta, dtype=y_dtype)[()]
    jacobian = np.zeros((y_size, x_size), dtype=x_dtype)
    f = _prepare(f, xs_dtypes, xs_shapes)
    for col in range(x_size):
        original = x.ravel().view(x_dtype)[col]
        x.ravel().view(x_dtype)[col] += delta
        y_pos = _to_numpy(f(*xs))
        x.ravel().view(x_dtype)[col] = original
        x.ravel().view(x_dtype)[col] -= delta
        y_neg = _to_numpy(f(*xs))
        x.ravel().view(x_dtype)[col] = original
        diff = (y_pos - y_neg) / scale
        jacobian[:, col] = diff.ravel().view(y_dtype)
    logging.vlog(1, 'Numeric Jacobian =\n%s', jacobian)
    return jacobian