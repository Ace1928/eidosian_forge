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
class DTensorDistributedValue(values_lib.DistributedValues):
    """DistributedValue backed by a DTensor instance.

  This class is useful to align the interface between DTensor and tf.distribute.
  Most of the tf.distribute API will accept/return DistributedValue, whereas
  DTensor low level API will only accept DTensor instance. In order to avoid
  the conversion back and forth between DistributedValue and DTensor, we
  introduce this class so that it can work with both side.
  """

    def __init__(self, dtensor):
        if context.executing_eagerly():
            if not d_api.is_dtensor(dtensor):
                raise ValueError(f'The DTensorDistributedValue can only be built with DTensor instance, got {type(dtensor)}')
            super().__init__(d_api.unpack(dtensor))
        else:
            super().__init__([dtensor])
        self._dtensor = dtensor

    def get_dtensor(self):
        return self._dtensor

    @property
    def values(self):
        return self._values