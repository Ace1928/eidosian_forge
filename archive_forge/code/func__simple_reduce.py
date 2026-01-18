import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _simple_reduce(per_replica_value, reduce_to_device, accumulation_fn, reduce_op):
    """Reduces the value by accumulation_fn and reduce_op."""
    all_values = per_replica_value.values
    if not all_values:
        raise ValueError('`per_replica_value` must be non-empty')
    count = len(all_values)
    with ops.device(reduce_to_device):
        with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            reduced = cross_device_utils.aggregate_tensors_or_indexed_slices(all_values, accumulation_fn)
            if reduce_op == reduce_util.ReduceOp.MEAN:
                reduced = cross_device_utils.divide_by_n_tensors_or_indexed_slices(reduced, count)
            elif reduce_op != reduce_util.ReduceOp.SUM:
                raise ValueError('`reduce_op` must be Reduce.SUM or Reduce.MEAN.')
    return reduced