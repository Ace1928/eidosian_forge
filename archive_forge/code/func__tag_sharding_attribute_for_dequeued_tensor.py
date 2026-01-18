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
def _tag_sharding_attribute_for_dequeued_tensor(tensor, dims):
    """Tags appropriate XLA sharding attribute to the dequeued tensor.

  The sharding attribute of the dequeued tensor will be a tuple.

  Args:
    tensor: The dequeued tensor on TPU.
    dims: A list of integer describes how the tensor is partitioned.

  Returns:
    The same tensor with the xla_sharding attribute.
  """
    if dims is None:
        return xla_sharding.replicate(tensor, assign_tuple_sharding=True)
    elif np.prod(dims) == 1:
        return xla_sharding.assign_device(tensor, 0, assign_tuple_sharding=True)
    else:
        tile_assignment = np.arange(np.prod(dims)).reshape(dims)
        return xla_sharding.tile(tensor=tensor, tile_assignment=tile_assignment, assign_tuple_sharding=True)