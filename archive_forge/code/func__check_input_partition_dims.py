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
def _check_input_partition_dims(self, tensor, dims):
    """Checks that input partition dims are valid for the `Tensor`.

    Args:
      tensor: Input tensor for partitioning.
      dims: A list of integer describes how to partition the input tensor.

    Raises:
      ValueError: If the tensor can't be partitioned by dims or the
        num_cores_per_replica doesn't match the number of
        partitions(dims.prod()).
    """
    if dims is None:
        return
    dims = np.array(dims)
    if (dims < 1).any():
        raise ValueError('All input partition dims must be >= 1.')
    if dims.prod() == 1:
        return
    if dims.prod() != self._device_assignment.num_cores_per_replica:
        raise ValueError('The product of each input partition dim should equal to num_cores_per_replica. (dim = {}, num_cores_per_replica = {})'.format(dims, self._device_assignment.num_cores_per_replica))
    if dims.shape[0] != tensor.shape.ndims:
        raise ValueError('Input partition dims must have the same number of dimensions as the `Tensor` to be partitioned. (tensor shape = {}, input partition dims = {}).'.format(tensor.shape.as_list(), dims))
    tensor.shape.assert_is_fully_defined()