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
@tf_export('distribute.NcclAllReduce')
class NcclAllReduce(AllReduceCrossDeviceOps):
    """NCCL all-reduce implementation of CrossDeviceOps.

  It uses Nvidia NCCL for all-reduce. For the batch API, tensors will be
  repacked or aggregated for more efficient cross-device transportation.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.

  Here is how you can use `NcclAllReduce` in `tf.distribute.MirroredStrategy`:


  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.NcclAllReduce())
  ```
  """

    def __init__(self, num_packs=1):
        """Initializes the object.

    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.

    Raises:
      ValueError: if `num_packs` is negative.
    """
        if num_packs < 0:
            raise ValueError('NCCL all-reduce requires num_packs >= 0, but {} is specified'.format(num_packs))
        super(NcclAllReduce, self).__init__(all_reduce_alg='nccl', num_packs=num_packs)