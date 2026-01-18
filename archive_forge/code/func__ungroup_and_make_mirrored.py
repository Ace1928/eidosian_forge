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
def _ungroup_and_make_mirrored(grouped_reduced, destinations, reduce_op, num_between_graph_workers=1):
    """Ungroup results from all-reduce and make Mirrored objects.

  Each all-reduce result will be divided by the number of destinations before
  Mirrored objects are created if reduce_op is "mean".

  Args:
    grouped_reduced: a list of lists, each sublist has components for each
      device, paired with a None. It is the result from
      cross_device_utils.aggregate_gradients_using*.
    destinations: a value to colocate the result with.
    reduce_op: Indicates how values will be aggregated. Accepted values
      are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
    num_between_graph_workers: number of workers in the between-graph
      replication.

  Returns:
    a list of Mirrored objects.
  """
    num_replicas = len(get_devices_from(destinations)) * num_between_graph_workers
    index = [[] for _ in range(len(grouped_reduced[0]))]
    for per_replica_reduced in grouped_reduced:
        for i, (v, _) in enumerate(per_replica_reduced):
            if reduce_op == reduce_util.ReduceOp.MEAN:
                with ops.device(v.device):
                    index[i].append(v / num_replicas)
            else:
                index[i].append(v)
    return [distribute_utils.regroup(v, wrap_class=value_lib.Mirrored) for v in index]