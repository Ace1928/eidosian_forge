from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
def _tensor_conversion_packed_var_and_device(var, dtype=None, name=None, as_ref=False):
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)