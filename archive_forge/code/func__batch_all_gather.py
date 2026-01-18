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
def _batch_all_gather(self, per_replica_values, axis, options):
    """all gather multiple per-replica-values."""
    batch_size = len(per_replica_values)
    if self._limited_nccl and options.implementation == collective_util.CommunicationImplementation.NCCL and (batch_size == 1):
        options = options.merge(collective_util.Options(implementation=collective_util.CommunicationImplementation.RING))
    logging.log_first_n(logging.INFO, 'Collective batch_all_gather: %d all-gathers, num_devices = %d, group_size = %d, implementation = %s, ' % (batch_size, len(self._devices), self._group_size, options.implementation), 10)

    def compute_gathered_values():
        gathered_values = []
        with self._lock, ops.name_scope('allgather'):
            for per_replica in per_replica_values:
                outputs = []
                for i in range(len(self._devices)):
                    outputs.append(self._launchers[i].all_gather(per_replica.values[i], axis, options))
                gathered_values.append(outputs)
        return gathered_values
    if context.executing_eagerly():
        gathered_values = def_function.function(compute_gathered_values)()
    else:
        gathered_values = compute_gathered_values()
    mirrored = []
    for value in gathered_values:
        mirrored.append(distribute_utils.regroup(value, wrap_class=value_lib.Mirrored))
    return mirrored