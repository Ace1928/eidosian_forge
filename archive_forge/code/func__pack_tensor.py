import threading
import weakref
from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _pack_tensor(self, *tensors):
    """Helper to pack plain-old-tensors, not structures or composites."""
    for tensor in tensors:
        if not isinstance(tensor, (tensor_lib.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
            raise ValueError('Every component to pack onto the ParallelDevice must already be a tensor, got {}. Consider running `tf.constant` or `tf.convert_to_tensor` first on literal values.'.format(tensors))
    with ops.device(self._name):
        return tpu_ops.tpu_replicated_input(inputs=tensors)