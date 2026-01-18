from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
@custom_gradient.custom_gradient
def _identity_with_grad_check(x):
    """Function that asserts it's gradient has a certain value."""
    x = array_ops.identity(x)

    def grad(dx):
        """Gradient function that asserts the gradient has a certain value."""
        if expected_dtype:
            assert dx.dtype == expected_dtype, 'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype)
        expected_tensor = tensor_conversion.convert_to_tensor_v2_with_dispatch(expected_gradient, dtype=dx.dtype, name='expected_gradient')
        with ops.control_dependencies([x]):
            assert_op = check_ops.assert_equal(dx, expected_tensor)
        with ops.control_dependencies([assert_op]):
            dx = array_ops.identity(dx)
        return dx
    return (x, grad)