import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_shuffle_scatter(reduced_shards, dst_devices):
    """Build the scatter phase of shuffle all-reduce.

  Args:
    reduced_shards:  list of `tf.Tensor` fully reduced shards
    dst_devices: list of names of devices at which the fully-reduced value
      should be reconstituted.

  Returns:
    list of `tf.Tensor` scattered tensors.
  """
    num_devices = len(dst_devices)
    out_tensors = []
    for d in range(0, num_devices):
        with ops.device(dst_devices[d]):
            out_tensors.append(array_ops.concat(reduced_shards, 0))
    return out_tensors