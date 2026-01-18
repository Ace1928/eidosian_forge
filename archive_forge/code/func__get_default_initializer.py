import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
    del shape
    if dtype.is_floating:
        initializer = init_ops.glorot_uniform_initializer()
        initializing_from_value = False
    elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool or (dtype == dtypes.string):
        initializer = init_ops.zeros_initializer()
        initializing_from_value = False
    else:
        raise ValueError('An initializer for variable %s of %s is required' % (name, dtype.base_dtype))
    return (initializer, initializing_from_value)