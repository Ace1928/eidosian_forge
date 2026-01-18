import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_shuffle_hybrid(input_tensors, gather_devices, red_op, upper_level_f):
    """Construct a subgraph for Shuffle hybrid all-reduce.

  Args:
    input_tensors: list of `tf.Tensor` of same-shape and type values to
      be reduced.
    gather_devices: list of device names on which to host gather shards.
    red_op: binary elementwise reduction operator.
    upper_level_f: function for reducing one value per worker, across
      workers.

  Returns:
    list of `tf.Tensor` of reduced values.

  Raises:
    ValueError: inputs not well-formed.
  """
    input_tensors, shape = _flatten_tensors(input_tensors)
    devices = [t.device for t in input_tensors]
    per_worker_devices, per_worker_values = _split_by_task(devices, input_tensors)
    num_workers = len(per_worker_devices)
    up_values = []
    if len(gather_devices) != num_workers:
        raise ValueError('For shuffle hybrid, gather_devices must contain one device per worker. ')
    for w in range(0, num_workers):
        reduced_shards = _build_shuffle_gather(per_worker_values[w], [gather_devices[w]], red_op)
        up_values.append(reduced_shards[0])
    level_2_output = upper_level_f(up_values)
    output_tensors = []
    for w in range(0, num_workers):
        output_tensors += _build_shuffle_scatter([level_2_output[w]], per_worker_devices[w])
    if len(shape) != 1:
        output_tensors = _reshape_tensors(output_tensors, shape)
    return output_tensors