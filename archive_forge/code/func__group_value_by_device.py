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
def _group_value_by_device(per_replica_values):
    """Group values into sublists by their devices.

  This grouping is needed to call the all-reduce library because it expects a
  list of the following form:
    [[(grad0_gpu0, v0_gpu0), (grad1_gpu0, v1_gpu0), (grad2_gpu0, v2_gpu0) ...],
     [(grad0_gpu1, v0_gpu1), (grad1_gpu1, v1_gpu1), (grad2_gpu1, v2_gpu1) ...],
     [(grad0_gpu2, v0_gpu2), (grad1_gpu0, v1_gpu2), (grad2_gpu0, v2_gpu2) ...],
     ...
    ]

  Args:
    per_replica_values: a list of PerReplica objects.

  Returns:
    a list of lists, each sublist has components for its corresponding device of
      PerReplica objects, paired with a None.
  """
    destinations = per_replica_values[0]._devices
    grouped = [[] for _ in range(len(destinations))]
    for per_replica_value in per_replica_values:
        for i, v in enumerate(per_replica_value.values):
            assert per_replica_value._devices == destinations
            grouped[i].append((v, None))
    return grouped