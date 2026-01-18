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
def _do_batch_all_reduce(self, reduce_op, dense_values):
    """Run batch all-reduces."""
    logging.log_first_n(logging.INFO, 'batch_all_reduce: %d all-reduces with algorithm = %s, num_packs = %d' % (len(dense_values), self._all_reduce_alg, self._num_packs), 10)
    destinations = dense_values[0]._devices
    grouped = _group_value_by_device(dense_values)
    device_grad_packs, tensor_packer = _pack_tensors(grouped, self._num_packs)
    if self._all_reduce_alg == 'nccl':
        reduced = cross_device_utils.aggregate_gradients_using_nccl(device_grad_packs)
    else:
        reduced = cross_device_utils.aggregate_gradients_using_hierarchical_copy(destinations, device_grad_packs)
    reduced = _unpack_tensors(reduced, tensor_packer)
    return _ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)