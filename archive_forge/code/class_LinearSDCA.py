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
@estimator_export('estimator.experimental.LinearSDCA')
class LinearSDCA(object):
    """Stochastic Dual Coordinate Ascent helper for linear estimators.

  Objects of this class are intended to be provided as the optimizer argument
  (though LinearSDCA objects do not implement the `tf.train.Optimizer`
  interface)
  when creating `tf.estimator.LinearClassifier` or
  `tf.estimator.LinearRegressor`.

  SDCA can only be used with `LinearClassifier` and `LinearRegressor` under the
  following conditions:

    - Feature columns are of type V2.
    - Multivalent categorical columns are not normalized. In other words the
      `sparse_combiner` argument in the estimator constructor should be "sum".
    - For classification: binary label.
    - For regression: one-dimensional label.

  Example usage:

  ```python
  real_feature_column = numeric_column(...)
  sparse_feature_column = categorical_column_with_hash_bucket(...)
  linear_sdca = tf.estimator.experimental.LinearSDCA(
      example_id_column='example_id',
      num_loss_partitions=1,
      num_table_shards=1,
      symmetric_l2_regularization=2.0)
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[real_feature_column, sparse_feature_column],
      weight_column=...,
      optimizer=linear_sdca)
  classifier.train(input_fn_train, steps=50)
  classifier.evaluate(input_fn=input_fn_eval)
  ```

  Here the expectation is that the `input_fn_*` functions passed to train and
  evaluate return a pair (dict, label_tensor) where dict has `example_id_column`
  as `key` whose value is a `Tensor` of shape [batch_size] and dtype string.
  num_loss_partitions defines sigma' in eq (11) of [3]. Convergence of (global)
  loss is guaranteed if `num_loss_partitions` is larger or equal to the product
  `(#concurrent train ops/per worker) x (#workers)`. Larger values for
  `num_loss_partitions` lead to slower convergence. The recommended value for
  `num_loss_partitions` in `tf.estimator` (where currently there is one process
  per worker) is the number of workers running the train steps. It defaults to 1
  (single machine).
  `num_table_shards` defines the number of shards for the internal state
  table, typically set to match the number of parameter servers for large
  data sets.

  The SDCA algorithm was originally introduced in [1] and it was followed by
  the L1 proximal step [2], a distributed version [3] and adaptive sampling [4].
  [1] www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
  [2] https://arxiv.org/pdf/1309.2375.pdf
  [3] https://arxiv.org/pdf/1502.03508.pdf
  [4] https://arxiv.org/pdf/1502.08053.pdf
  Details specific to this implementation are provided in:
  https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/canned/linear_optimizer/doc/sdca.ipynb
  """

    def __init__(self, example_id_column, num_loss_partitions=1, num_table_shards=None, symmetric_l1_regularization=0.0, symmetric_l2_regularization=1.0, adaptive=False):
        """Construct a new SDCA optimizer for linear estimators.

    Args:
      example_id_column: The column name containing the example ids.
      num_loss_partitions: Number of workers.
      num_table_shards: Number of shards of the internal state table, typically
        set to match the number of parameter servers.
      symmetric_l1_regularization: A float value, must be greater than or equal
        to zero.
      symmetric_l2_regularization: A float value, must be greater than zero and
        should typically be greater than 1.
      adaptive: A boolean indicating whether to use adaptive sampling.
    """
        self._example_id_column = example_id_column
        self._num_loss_partitions = num_loss_partitions
        self._num_table_shards = num_table_shards
        self._symmetric_l1_regularization = symmetric_l1_regularization
        self._symmetric_l2_regularization = symmetric_l2_regularization
        self._adaptive = adaptive

    def _prune_and_unique_sparse_ids(self, id_weight_pair):
        """Remove duplicate and negative ids in a sparse tendor."""
        id_tensor = id_weight_pair.id_tensor
        if id_weight_pair.weight_tensor:
            weight_tensor = id_weight_pair.weight_tensor.values
        else:
            weight_tensor = tf.ones([tf.compat.v1.shape(id_tensor.indices)[0]], tf.dtypes.float32)
        example_ids = tf.reshape(id_tensor.indices[:, 0], [-1])
        flat_ids = tf.cast(tf.reshape(id_tensor.values, [-1]), dtype=tf.dtypes.int64)
        is_id_valid = tf.math.greater_equal(flat_ids, 0)
        flat_ids = tf.compat.v1.boolean_mask(flat_ids, is_id_valid)
        example_ids = tf.compat.v1.boolean_mask(example_ids, is_id_valid)
        weight_tensor = tf.compat.v1.boolean_mask(weight_tensor, is_id_valid)
        projection_length = tf.math.reduce_max(flat_ids) + 1
        projected_ids = projection_length * example_ids + flat_ids
        ids, idx = tf.unique(projected_ids)
        example_ids_filtered = tf.math.unsorted_segment_min(example_ids, idx, tf.compat.v1.shape(ids)[0])
        reproject_ids = ids - projection_length * example_ids_filtered
        weights = tf.reshape(tf.math.unsorted_segment_sum(weight_tensor, idx, tf.compat.v1.shape(ids)[0]), [-1])
        return sdca_ops._SparseFeatureColumn(example_ids_filtered, reproject_ids, weights)

    def get_train_step(self, state_manager, weight_column_name, loss_type, feature_columns, features, targets, bias_var, global_step):
        """Returns the training operation of an SdcaModel optimizer."""
        batch_size = tf.compat.v1.shape(targets)[0]
        cache = tf.compat.v2.__internal__.feature_column.FeatureTransformationCache(features)
        dense_features, dense_feature_weights = ([], [])
        sparse_feature_with_values, sparse_feature_with_values_weights = ([], [])
        for column in sorted(feature_columns, key=lambda x: x.name):
            if isinstance(column, feature_column_lib.CategoricalColumn):
                id_weight_pair = column.get_sparse_tensors(cache, state_manager)
                sparse_feature_with_values.append(self._prune_and_unique_sparse_ids(id_weight_pair))
                sparse_feature_with_values_weights.append(state_manager.get_variable(column, 'weights'))
            elif isinstance(column, tf.compat.v2.__internal__.feature_column.DenseColumn):
                if column.variable_shape.ndims != 1:
                    raise ValueError('Column %s has rank %d, larger than 1.' % (type(column).__name__, column.variable_shape.ndims))
                dense_features.append(column.get_dense_tensor(cache, state_manager))
                dense_feature_weights.append(state_manager.get_variable(column, 'weights'))
            else:
                raise ValueError('LinearSDCA does not support column type %s.' % type(column).__name__)
        dense_features.append(tf.ones([batch_size, 1]))
        dense_feature_weights.append(bias_var)
        example_weights = tf.reshape(features[weight_column_name], shape=[-1]) if weight_column_name else tf.ones([batch_size])
        example_ids = features[self._example_id_column]
        training_examples = dict(sparse_features=sparse_feature_with_values, dense_features=dense_features, example_labels=tf.compat.v1.to_float(tf.reshape(targets, shape=[-1])), example_weights=example_weights, example_ids=example_ids)
        training_variables = dict(sparse_features_weights=sparse_feature_with_values_weights, dense_features_weights=dense_feature_weights)
        sdca_model = sdca_ops._SDCAModel(examples=training_examples, variables=training_variables, options=dict(symmetric_l1_regularization=self._symmetric_l1_regularization, symmetric_l2_regularization=self._symmetric_l2_regularization, adaptive=self._adaptive, num_loss_partitions=self._num_loss_partitions, num_table_shards=self._num_table_shards, loss_type=loss_type))
        train_op = sdca_model.minimize(global_step=global_step)
        return (sdca_model, train_op)