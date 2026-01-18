import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('distribute.RunOptions')
class RunOptions(collections.namedtuple('RunOptions', ['experimental_enable_dynamic_batch_size', 'experimental_bucketizing_dynamic_shape', 'experimental_xla_options'])):
    """Run options for `strategy.run`.

  This can be used to hold some strategy specific configs.

  Attributes:
    experimental_enable_dynamic_batch_size: Boolean. Only applies to
      TPUStrategy. Default to True. If True, TPUStrategy will enable dynamic
      padder to support dynamic batch size for the inputs. Otherwise only static
      shape inputs are allowed.
    experimental_bucketizing_dynamic_shape: Boolean. Only applies to
      TPUStrategy. Default to False. If True, TPUStrategy will automatic
      bucketize inputs passed into `run` if the input shape is
      dynamic. This is a performance optimization to reduce XLA recompilation,
      which should not have impact on correctness.
    experimental_xla_options: A `tf.tpu.XLAOptions` instance. Only applies to
      TPUStrategy. Controls the XLA compiling options on TPUs. Default to None.
  """

    def __new__(cls, experimental_enable_dynamic_batch_size=True, experimental_bucketizing_dynamic_shape=False, experimental_xla_options=None):
        return super(RunOptions, cls).__new__(cls, experimental_enable_dynamic_batch_size, experimental_bucketizing_dynamic_shape, experimental_xla_options)