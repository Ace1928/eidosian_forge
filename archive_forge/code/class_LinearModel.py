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
class LinearModel(tf.keras.Model):
    """Produces a linear prediction `Tensor` based on given `feature_columns`.

  This layer generates a weighted sum based on output dimension `units`.
  Weighted sum refers to logits in classification problems. It refers to the
  prediction itself for linear regression problems.

  Note on supported columns: `LinearLayer` treats categorical columns as
  `indicator_column`s. To be specific, assume the input as `SparseTensor` looks
  like:

  ```python
    shape = [2, 2]
    {
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
    }
  ```
  `linear_model` assigns weights for the presence of "a", "b", "c' implicitly,
  just like `indicator_column`, while `input_layer` explicitly requires wrapping
  each of categorical columns with an `embedding_column` or an
  `indicator_column`.

  Example of usage:

  ```python
  price = numeric_column('price')
  price_buckets = bucketized_column(price, boundaries=[0., 10., 100., 1000.])
  keywords = categorical_column_with_hash_bucket("keywords", 10K)
  keywords_price = crossed_column('keywords', price_buckets, ...)
  columns = [price_buckets, keywords, keywords_price ...]
  linear_model = LinearLayer(columns)

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  prediction = linear_model(features)
  ```
  """

    def __init__(self, feature_columns, units=1, sparse_combiner='sum', trainable=True, name=None, **kwargs):
        """Constructs a LinearLayer.

    Args:
      feature_columns: An iterable containing the FeatureColumns to use as
        inputs to your model. All items should be instances of classes derived
        from `_FeatureColumn`s.
      units: An integer, dimensionality of the output space. Default value is 1.
      sparse_combiner: A string specifying how to reduce if a categorical column
        is multivalent. Except `numeric_column`, almost all columns passed to
        `linear_model` are considered as categorical columns.  It combines each
        categorical column independently. Currently "mean", "sqrtn" and "sum"
        are supported, with "sum" the default for linear model. "sqrtn" often
        achieves good accuracy, in particular with bag-of-words columns.
          * "sum": do not normalize features in the column
          * "mean": do l1 normalization on features in the column
          * "sqrtn": do l2 normalization on features in the column
        For example, for two features represented as the categorical columns:

          ```python
          # Feature 1

          shape = [2, 2]
          {
              [0, 0]: "a"
              [0, 1]: "b"
              [1, 0]: "c"
          }

          # Feature 2

          shape = [2, 3]
          {
              [0, 0]: "d"
              [1, 0]: "e"
              [1, 1]: "f"
              [1, 2]: "g"
          }
          ```

        with `sparse_combiner` as "mean", the linear model outputs conceptually
        are
        ```
        y_0 = 1.0 / 2.0 * ( w_a + w_ b) + w_c + b_0
        y_1 = w_d + 1.0 / 3.0 * ( w_e + w_ f + w_g) + b_1
        ```
        where `y_i` is the output, `b_i` is the bias, and `w_x` is the weight
        assigned to the presence of `x` in the input features.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: Name to give to the Linear Model. All variables and ops created will
        be scoped by this name.
      **kwargs: Keyword arguments to construct a layer.

    Raises:
      ValueError: if an item in `feature_columns` is neither a `DenseColumn`
        nor `CategoricalColumn`.
    """
        super(LinearModel, self).__init__(name=name, **kwargs)
        self.layer = _LinearModelLayer(feature_columns, units, sparse_combiner, trainable, name=self.name, **kwargs)

    def call(self, features):
        """Returns a `Tensor` the represents the predictions of a linear model.

    Args:
      features: A mapping from key to tensors. `_FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values are `Tensor` or `SparseTensor` depending on
        corresponding `_FeatureColumn`.

    Returns:
      A `Tensor` which represents predictions/logits of a linear model. Its
      shape is (batch_size, units) and its dtype is `float32`.

    Raises:
      ValueError: If features are not a dictionary.
    """
        return self.layer(features)

    @property
    def bias(self):
        return self.layer.bias