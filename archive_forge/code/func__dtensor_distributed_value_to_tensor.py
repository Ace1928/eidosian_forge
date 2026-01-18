from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
def _dtensor_distributed_value_to_tensor(var, dtype=None, name=None, as_ref=False):
    del name
    dtensor = var.get_dtensor()
    if dtype is not None and (not dtype.is_compatible_with(dtensor.dtype)):
        raise ValueError('Incompatible type conversion requested to type {!r} for variable of type {!r}'.format(dtype.name, dtensor.dtype.name))
    if as_ref:
        raise NotImplementedError("PerReplica doesn't support being used as a reference.")
    return dtensor