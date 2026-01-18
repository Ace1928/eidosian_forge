import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _weighted_categorical_column(categorical_column, weight_feature_key, dtype=dtypes.float32):
    """Applies weight values to a `_CategoricalColumn`.

  Use this when each of your sparse inputs has both an ID and a value. For
  example, if you're representing text documents as a collection of word
  frequencies, you can provide 2 parallel sparse input features ('terms' and
  'frequencies' below).

  Example:

  Input `tf.Example` objects:

  ```proto
  [
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "very" value: "model"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.3 value: 0.1}}
      }
    },
    features {
      feature {
        key: "terms"
        value {bytes_list {value: "when" value: "course" value: "human"}}
      }
      feature {
        key: "frequencies"
        value {float_list {value: 0.4 value: 0.1 value: 0.2}}
      }
    }
  ]
  ```

  ```python
  categorical_column = categorical_column_with_hash_bucket(
      column_name='terms', hash_bucket_size=1000)
  weighted_column = weighted_categorical_column(
      categorical_column=categorical_column, weight_feature_key='frequencies')
  columns = [weighted_column, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  This assumes the input dictionary contains a `SparseTensor` for key
  'terms', and a `SparseTensor` for key 'frequencies'. These 2 tensors must have
  the same indices and dense shape.

  Args:
    categorical_column: A `_CategoricalColumn` created by
      `categorical_column_with_*` functions.
    weight_feature_key: String key for weight values.
    dtype: Type of weights, such as `tf.float32`. Only float and integer weights
      are supported.

  Returns:
    A `_CategoricalColumn` composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if `dtype` is not convertible to float.
  """
    if dtype is None or not (dtype.is_integer or dtype.is_floating):
        raise ValueError('dtype {} is not convertible to float.'.format(dtype))
    return _WeightedCategoricalColumn(categorical_column=categorical_column, weight_feature_key=weight_feature_key, dtype=dtype)