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
def _batch_all_reduce(self, reduce_op, per_replica_values):
    """All-reduce algorithm in a batch."""
    dense_values, dense_indices, sparse_values, sparse_indices = cross_device_utils.split_by_sparsity(per_replica_values)
    if dense_values:
        dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
    else:
        dense_results = []
    if sparse_values:
        sparse_results = self._do_batch_all_reduce_sparse(reduce_op, sparse_values)
    else:
        sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices), (sparse_results, sparse_indices)))