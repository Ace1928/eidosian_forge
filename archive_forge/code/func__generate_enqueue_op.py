import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def _generate_enqueue_op(self, inputs, name_prefix, index, device=None, tpu_ordinal=-1):
    """Generate a host-side Op to enqueue a tuple to the queue.

    If device is None the inputs are all required to have the same
    device specification, and the enqueue Op is colocated with
    inputs[0]. Otherwise the enqueue Op is placed on 'device'.

    Args:
      inputs: a list of Tensors with the types and shapes of the tuple elements.
      name_prefix: the base name for the Op.
      index: the shard index, used to uniquify the Op name.
      device: device to place the Op on, or None if it should be
        colocated with the inputs.
      tpu_ordinal: ordinal of the TPU device on the host to use for
      infeed if device is a CPU device. Should be set to -1 if device
      is a TPU device.

    Returns:
      An Op corresponding to a shard of infeed enqueued at the host,
      suitable for use within a replicated block.

    Raises:
      ValueError: if device is None and inputs do not all have the
        same device specification.
    """
    full_name = '%s/%d' % (name_prefix, index)
    shapes = [t.shape for t in inputs]
    if device is None:
        devices = [t.device for t in inputs]
        for i in range(1, self.number_of_tuple_elements):
            if devices[0] != devices[i]:
                raise ValueError(f'input devices for shard {index} are {str(devices)}, but should all be the same')
        with ops.colocate_with(inputs[0]):
            return tpu_ops.infeed_enqueue_tuple(inputs=inputs, shapes=shapes, name=full_name, device_ordinal=tpu_ordinal)
    else:
        with ops.device(device):
            return tpu_ops.infeed_enqueue_tuple(inputs=inputs, shapes=shapes, name=full_name, device_ordinal=tpu_ordinal)