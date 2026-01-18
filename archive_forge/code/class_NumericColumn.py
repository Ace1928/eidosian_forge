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
@serialization.register_feature_column
class NumericColumn(DenseColumn, fc_old._DenseColumn, collections.namedtuple('NumericColumn', ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
    """see `numeric_column`."""

    @property
    def _is_v2_column(self):
        return True

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return self.key

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return {self.key: parsing_ops.FixedLenFeature(self.shape, self.dtype, self.default_value)}

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.parse_example_spec

    def _transform_input_tensor(self, input_tensor):
        if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError('The corresponding Tensor of numerical column must be a Tensor. SparseTensor is not supported. key: {}'.format(self.key))
        if self.normalizer_fn is not None:
            input_tensor = self.normalizer_fn(input_tensor)
        return math_ops.cast(input_tensor, dtypes.float32)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        return self._transform_input_tensor(input_tensor)

    def transform_feature(self, transformation_cache, state_manager):
        """See `FeatureColumn` base class.

    In this case, we apply the `normalizer_fn` to the input tensor.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Normalized input tensor.
    Raises:
      ValueError: If a SparseTensor is passed in.
    """
        input_tensor = transformation_cache.get(self.key, state_manager)
        return self._transform_input_tensor(input_tensor)

    @property
    def variable_shape(self):
        """See `DenseColumn` base class."""
        return tensor_shape.TensorShape(self.shape)

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _variable_shape(self):
        return self.variable_shape

    def get_dense_tensor(self, transformation_cache, state_manager):
        """Returns dense `Tensor` representing numeric feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.
    """
        return transformation_cache.get(self, state_manager)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        return inputs.get(self)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.key]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        config = dict(zip(self._fields, self))
        config['normalizer_fn'] = serialization._serialize_keras_object(self.normalizer_fn)
        config['dtype'] = self.dtype.name
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['normalizer_fn'] = serialization._deserialize_keras_object(config['normalizer_fn'], custom_objects=custom_objects)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        return cls(**kwargs)