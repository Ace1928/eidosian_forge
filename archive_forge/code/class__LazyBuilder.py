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
class _LazyBuilder(object):
    """Handles caching of transformations while building the model.

  `_FeatureColumn` specifies how to digest an input column to the network. Some
  feature columns require data transformations. This class caches those
  transformations.

  Some features may be used in more than one place. For example, one can use a
  bucketized feature by itself and a cross with it. In that case we
  should create only one bucketization op instead of creating ops for each
  feature column separately. To handle re-use of transformed columns,
  `_LazyBuilder` caches all previously transformed columns.

  Example:
  We're trying to use the following `_FeatureColumn`s:

  ```python
  bucketized_age = fc.bucketized_column(fc.numeric_column("age"), ...)
  keywords = fc.categorical_column_with_hash_buckets("keywords", ...)
  age_X_keywords = fc.crossed_column([bucketized_age, "keywords"])
  ... = linear_model(features,
                          [bucketized_age, keywords, age_X_keywords]
  ```

  If we transform each column independently, then we'll get duplication of
  bucketization (one for cross, one for bucketization itself).
  The `_LazyBuilder` eliminates this duplication.
  """

    def __init__(self, features):
        """Creates a `_LazyBuilder`.

    Args:
      features: A mapping from feature column to objects that are `Tensor` or
        `SparseTensor`, or can be converted to same via
        `sparse_tensor.convert_to_tensor_or_sparse_tensor`. A `string` key
        signifies a base feature (not-transformed). A `_FeatureColumn` key means
        that this `Tensor` is the output of an existing `_FeatureColumn` which
        can be reused.
    """
        self._features = features.copy()
        self._feature_tensors = {}

    def get(self, key):
        """Returns a `Tensor` for the given key.

    A `str` key is used to access a base feature (not-transformed). When a
    `_FeatureColumn` is passed, the transformed feature is returned if it
    already exists, otherwise the given `_FeatureColumn` is asked to provide its
    transformed output, which is then cached.

    Args:
      key: a `str` or a `_FeatureColumn`.

    Returns:
      The transformed `Tensor` corresponding to the `key`.

    Raises:
      ValueError: if key is not found or a transformed `Tensor` cannot be
        computed.
    """
        if key in self._feature_tensors:
            return self._feature_tensors[key]
        if key in self._features:
            feature_tensor = self._get_raw_feature_as_tensor(key)
            self._feature_tensors[key] = feature_tensor
            return feature_tensor
        if isinstance(key, six.string_types):
            raise ValueError('Feature {} is not in features dictionary.'.format(key))
        if not isinstance(key, _FeatureColumn):
            raise TypeError('"key" must be either a "str" or "_FeatureColumn". Provided: {}'.format(key))
        column = key
        logging.debug('Transforming feature_column %s.', column)
        transformed = column._transform_feature(self)
        if transformed is None:
            raise ValueError('Column {} is not supported.'.format(column.name))
        self._feature_tensors[column] = transformed
        return transformed

    def _get_raw_feature_as_tensor(self, key):
        """Gets the raw_feature (keyed by `key`) as `tensor`.

    The raw feature is converted to (sparse) tensor and maybe expand dim.

    For both `Tensor` and `SparseTensor`, the rank will be expanded (to 2) if
    the rank is 1. This supports dynamic rank also. For rank 0 raw feature, will
    error out as it is not supported.

    Args:
      key: A `str` key to access the raw feature.

    Returns:
      A `Tensor` or `SparseTensor`.

    Raises:
      ValueError: if the raw feature has rank 0.
    """
        raw_feature = self._features[key]
        feature_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(raw_feature)

        def expand_dims(input_tensor):
            if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
                return sparse_ops.sparse_reshape(input_tensor, [array_ops.shape(input_tensor)[0], 1])
            else:
                return array_ops.expand_dims(input_tensor, -1)
        rank = feature_tensor.get_shape().ndims
        if rank is not None:
            if rank == 0:
                raise ValueError('Feature (key: {}) cannot have rank 0. Given: {}'.format(key, feature_tensor))
            return feature_tensor if rank != 1 else expand_dims(feature_tensor)
        with ops.control_dependencies([check_ops.assert_positive(array_ops.rank(feature_tensor), message='Feature (key: {}) cannot have rank 0. Given: {}'.format(key, feature_tensor))]):
            return cond.cond(math_ops.equal(1, array_ops.rank(feature_tensor)), lambda: expand_dims(feature_tensor), lambda: feature_tensor)