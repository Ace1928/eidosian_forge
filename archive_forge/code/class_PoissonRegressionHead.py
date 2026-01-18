from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.PoissonRegressionHead')
class PoissonRegressionHead(RegressionHead):
    """Creates a `Head` for poisson regression using `tf.nn.log_poisson_loss`.

  The loss is the weighted sum over all input dimensions. Namely, if the input
  labels have shape `[batch_size, label_dimension]`, the loss is the weighted
  sum over both `batch_size` and `label_dimension`.

  The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`.
  In many applications, the shape is `[batch_size, label_dimension]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
  `[D0, D1, ... DN]` is also supported.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
  `[D0, D1, ... DN, label_dimension]`.

  This is implemented as a generalized linear model, see
  https://en.wikipedia.org/wiki/Generalized_linear_model.

  The head can be used with a canned estimator. Example:

  ```python
  my_head = tf.estimator.PoissonRegressionHead()
  my_estimator = tf.estimator.DNNEstimator(
      head=my_head,
      hidden_units=...,
      feature_columns=...)
  ```

  It can also be used with a custom `model_fn`. Example:

  ```python
  def _my_model_fn(features, labels, mode):
    my_head = tf.estimator.PoissonRegressionHead()
    logits = tf.keras.Model(...)(features)

    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.1),
        logits=logits)

  my_estimator = tf.estimator.Estimator(model_fn=_my_model_fn)
  ```

  Args:
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Decides how to
      reduce training loss over batch and label dimension. Defaults to
      `SUM_OVER_BATCH_SIZE`, namely weighted sum of losses divided by `batch
      size * label_dimension`.
    compute_full_loss: Whether to include the constant `log(z!)` term in
      computing the poisson loss. See `tf.nn.log_poisson_loss` for the full
      documentation.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.
  """

    def __init__(self, label_dimension=1, weight_column=None, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, compute_full_loss=True, name=None):
        self._compute_full_loss = compute_full_loss
        super(PoissonRegressionHead, self).__init__(label_dimension=label_dimension, weight_column=weight_column, loss_reduction=loss_reduction, loss_fn=self._poisson_loss, inverse_link_fn=tf.math.exp, name=name)

    def _poisson_loss(self, labels, logits):
        return tf.nn.log_poisson_loss(targets=labels, log_input=logits, compute_full_loss=self._compute_full_loss)