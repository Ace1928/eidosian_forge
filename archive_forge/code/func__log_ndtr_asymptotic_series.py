import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    dtype = x.dtype.as_numpy_dtype
    if series_order <= 0:
        return np.array(1, dtype)
    x_2 = math_ops.square(x)
    even_sum = array_ops.zeros_like(x)
    odd_sum = array_ops.zeros_like(x)
    x_2n = x_2
    for n in range(1, series_order + 1):
        y = np.array(_double_factorial(2 * n - 1), dtype) / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n *= x_2
    return 1.0 + even_sum - odd_sum