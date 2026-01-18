import abc
import functools
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls
def _get_reduction(self):
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if not self._allow_sum_over_batch_size and distribute_lib.has_strategy() and (self.reduction == losses_utils.ReductionV2.AUTO or self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
        raise ValueError('Please use `tf.keras.losses.Reduction.SUM` or `tf.keras.losses.Reduction.NONE` for loss reduction when losses are used with `tf.distribute.Strategy` outside of the built-in training loops. You can implement `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch size like:\n```\nwith strategy.scope():\n    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)\n....\n    loss = tf.reduce_sum(loss_obj(labels, predictions)) * (1. / global_batch_size)\n```\nPlease see https://www.tensorflow.org/tutorials/distribute/custom_training for more details.')
    if self.reduction == losses_utils.ReductionV2.AUTO:
        return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction