from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.LinearEstimator', v1=[])
class LinearEstimatorV2(estimator.EstimatorV2):
    """An estimator for TensorFlow linear models with user-specified head.

  Example:

  ```python
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = tf.estimator.LinearEstimator(
      head=tf.estimator.MultiLabelHead(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b])

  # Or estimator using an optimizer with a learning rate decay.
  estimator = tf.estimator.LinearEstimator(
      head=tf.estimator.MultiLabelHead(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b],
      optimizer=lambda: tf.keras.optimizers.Ftrl(
          learning_rate=tf.compat.v1.train.exponential_decay(
              learning_rate=0.1,
              global_step=tf.compat.v1.train.get_global_step(),
              decay_steps=10000,
              decay_rate=0.96))

  # Or estimator using the FTRL optimizer with regularization.
  estimator = tf.estimator.LinearEstimator(
      head=tf.estimator.MultiLabelHead(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b])
      optimizer=tf.keras.optimizers.Ftrl(
          learning_rate=0.1,
          l1_regularization_strength=0.001
      ))

  def input_fn_train:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_eval:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_predict:
    # Returns tf.data.Dataset of (x, None) tuple.
    pass
  estimator.train(input_fn=input_fn_train, steps=100)
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with `key=weight_column` whose
    value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss and predicted output are determined by the specified head.

  @compatibility(eager)
  Estimators can be used while eager execution is enabled. Note that `input_fn`
  and all hooks are executed inside a graph context, so they have to be written
  to be compatible with graph mode. Note that `input_fn` code using `tf.data`
  generally works in both graph and eager modes.
  @end_compatibility
  """

    def __init__(self, head, feature_columns, model_dir=None, optimizer='Ftrl', config=None, sparse_combiner='sum', warm_start_from=None):
        """Initializes a `LinearEstimator` instance.

    Args:
      head: A `Head` instance constructed with a method such as
        `tf.estimator.MultiLabelHead`.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.keras.optimizers.*` used to train the model.
        Can also be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp',
        'SGD'), or callable. Defaults to FTRL optimizer.
      config: `RunConfig` object to configure the runtime settings.
      sparse_combiner: A string specifying how to reduce if a categorical column
        is multivalent.  One of "mean", "sqrtn", and "sum" -- these are
        effectively different ways to do example-level normalization, which can
        be useful for bag-of-words features. for more details, see
        `tf.feature_column.linear_model`.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights and biases are warm-started, and it is assumed that vocabularies
        and Tensor names are unchanged.
    """

        def _model_fn(features, labels, mode, config):
            return _linear_model_fn_v2(features=features, labels=labels, mode=mode, head=head, feature_columns=tuple(feature_columns or []), optimizer=optimizer, config=config, sparse_combiner=sparse_combiner)
        estimator._canned_estimator_api_gauge.get_cell('Estimator').set('Linear')
        super(LinearEstimatorV2, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, warm_start_from=warm_start_from)