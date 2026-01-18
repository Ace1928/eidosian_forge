import abc
import collections
import math
import re
import numpy as np
import six
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _transform_features_v2(features, feature_columns, state_manager):
    """Returns transformed features based on features columns passed in.

  Please note that most probably you would not need to use this function. Please
  check `input_layer` and `linear_model` to see whether they will
  satisfy your use case or not.

  Example:

  ```python
  # Define features and transformations
  crosses_a_x_b = crossed_column(
      columns=["sparse_feature_a", "sparse_feature_b"], hash_bucket_size=10000)
  price_buckets = bucketized_column(
      source_column=numeric_column("price"), boundaries=[...])

  columns = [crosses_a_x_b, price_buckets]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  transformed = transform_features(features=features, feature_columns=columns)

  assertCountEqual(columns, transformed.keys())
  ```

  Args:
    features: A mapping from key to tensors. `FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values can be a `SparseTensor` or a `Tensor` depends on
      corresponding `FeatureColumn`.
    feature_columns: An iterable containing all the `FeatureColumn`s.
    state_manager: A StateManager object that holds the FeatureColumn state.

  Returns:
    A `dict` mapping `FeatureColumn` to `Tensor` and `SparseTensor` values.
  """
    feature_columns = _normalize_feature_columns(feature_columns)
    outputs = {}
    with ops.name_scope(None, default_name='transform_features', values=features.values()):
        transformation_cache = FeatureTransformationCache(features)
        for column in feature_columns:
            with ops.name_scope(None, default_name=_sanitize_column_name_for_variable_scope(column.name)):
                outputs[column] = transformation_cache.get(column, state_manager)
    return outputs