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
class Loss:
    """Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:

  ```python
  class MeanSquaredError(Loss):

    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.

  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.

  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:

  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """Initializes `Loss` class.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
    """
        losses_utils.ReductionV2.validate(reduction)
        self.reduction = reduction
        self.name = name
        self._allow_sum_over_batch_size = False
        self._set_name_scope()

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            self._name_scope = self.__class__.__name__
        elif self.name == '<lambda>':
            self._name_scope = 'lambda'
        else:
            self._name_scope = self.name.strip('_')

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      sample_weight: Optional `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `y_pred` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
        because all loss functions reduce by 1 dimension, usually axis=-1.)

    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(y_true, y_pred, sample_weight)
        with backend.name_scope(self._name_scope), graph_ctx:
            if context.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses = call_fn(y_true, y_pred)
            return losses_utils.compute_weighted_loss(losses, sample_weight, reduction=self._get_reduction())

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Loss` instance.
    """
        return cls(**config)

    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        return {'reduction': self.reduction, 'name': self.name}

    @abc.abstractmethod
    @doc_controls.for_subclass_implementers
    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
        raise NotImplementedError('Must be implemented in subclasses.')

    def _get_reduction(self):
        """Handles `AUTO` reduction cases and returns the reduction value."""
        if not self._allow_sum_over_batch_size and distribute_lib.has_strategy() and (self.reduction == losses_utils.ReductionV2.AUTO or self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
            raise ValueError('Please use `tf.keras.losses.Reduction.SUM` or `tf.keras.losses.Reduction.NONE` for loss reduction when losses are used with `tf.distribute.Strategy` outside of the built-in training loops. You can implement `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch size like:\n```\nwith strategy.scope():\n    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)\n....\n    loss = tf.reduce_sum(loss_obj(labels, predictions)) * (1. / global_batch_size)\n```\nPlease see https://www.tensorflow.org/tutorials/distribute/custom_training for more details.')
        if self.reduction == losses_utils.ReductionV2.AUTO:
            return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        return self.reduction