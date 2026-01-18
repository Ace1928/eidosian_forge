from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.BaselineClassifier', v1=[])
class BaselineClassifierV2(estimator.EstimatorV2):
    """A classifier that can establish a simple baseline.

  This classifier ignores feature values and will learn to predict the average
  value of each label. For single-label problems, this will predict the
  probability distribution of the classes as seen in the labels. For multi-label
  problems, this will predict the fraction of examples that are positive for
  each class.

  Example:

  ```python

  # Build BaselineClassifier
  classifier = tf.estimator.BaselineClassifier(n_classes=3)

  # Input builders
  def input_fn_train:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass

  def input_fn_eval:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass

  # Fit model.
  classifier.train(input_fn=input_fn_train)

  # Evaluate cross entropy between the test and train labels.
  loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

  # predict outputs the probability distribution of the classes as seen in
  # training.
  predictions = classifier.predict(new_samples)

  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
     `key=weight_column` whose value is a `Tensor`.

  @compatibility(eager)
  Estimators can be used while eager execution is enabled. Note that `input_fn`
  and all hooks are executed inside a graph context, so they have to be written
  to be compatible with graph mode. Note that `input_fn` code using `tf.data`
  generally works in both graph and eager modes.
  @end_compatibility
  """

    def __init__(self, model_dir=None, n_classes=2, weight_column=None, label_vocabulary=None, optimizer='Ftrl', config=None, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE):
        """Initializes a BaselineClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        It must be greater than 1. Note: Class labels are integers representing
          the class index (i.e. values from 0 to n_classes-1). For arbitrary
          label values (e.g. string labels), convert to class indices first.
      weight_column: A string or a `NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It will be multiplied by the loss of the example.
      label_vocabulary: Optional list of strings with size `[n_classes]`
        defining the label vocabulary. Only supported for `n_classes` > 2.
      optimizer: String, `tf.keras.optimizers.*` object, or callable that
        creates the optimizer to use for training. If not specified, will use
        `Ftrl` as the default optimizer.
      config: `RunConfig` object to configure the runtime settings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.

    Returns:
      A `BaselineClassifier` estimator.

    Raises:
      ValueError: If `n_classes` < 2.
    """
        head = head_utils.binary_or_multi_class_head(n_classes, weight_column=weight_column, label_vocabulary=label_vocabulary, loss_reduction=loss_reduction)

        def _model_fn(features, labels, mode, config):
            return _baseline_model_fn_v2(features=features, labels=labels, mode=mode, head=head, optimizer=optimizer, weight_column=weight_column, config=config, loss_reduction=loss_reduction)
        super(BaselineClassifierV2, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)