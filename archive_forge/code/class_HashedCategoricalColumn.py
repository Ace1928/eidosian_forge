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
class HashedCategoricalColumn(CategoricalColumn, fc_old._CategoricalColumn, collections.namedtuple('HashedCategoricalColumn', ('key', 'hash_bucket_size', 'dtype'))):
    """see `categorical_column_with_hash_bucket`."""

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
        return {self.key: parsing_ops.VarLenFeature(self.dtype)}

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.parse_example_spec

    def _transform_input_tensor(self, input_tensor):
        """Hashes the values in the feature_column."""
        if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError('SparseColumn input must be a SparseTensor.')
        fc_utils.assert_string_or_int(input_tensor.dtype, prefix='column_name: {} input_tensor'.format(self.key))
        if self.dtype.is_integer != input_tensor.dtype.is_integer:
            raise ValueError('Column dtype and SparseTensors dtype must be compatible. key: {}, column dtype: {}, tensor dtype: {}'.format(self.key, self.dtype, input_tensor.dtype))
        if self.dtype == dtypes.string:
            sparse_values = input_tensor.values
        else:
            sparse_values = string_ops.as_string(input_tensor.values)
        sparse_id_values = string_ops.string_to_hash_bucket_fast(sparse_values, self.hash_bucket_size, name='lookup')
        return sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

    def transform_feature(self, transformation_cache, state_manager):
        """Hashes the values in the feature_column."""
        input_tensor = _to_sparse_input_and_drop_ignore_values(transformation_cache.get(self.key, state_manager))
        return self._transform_input_tensor(input_tensor)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
        return self._transform_input_tensor(input_tensor)

    @property
    def num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.hash_bucket_size

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _num_buckets(self):
        return self.num_buckets

    def get_sparse_tensors(self, transformation_cache, state_manager):
        """See `CategoricalColumn` base class."""
        return CategoricalColumn.IdWeightPair(transformation_cache.get(self, state_manager), None)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        return CategoricalColumn.IdWeightPair(inputs.get(self), None)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.key]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        config = dict(zip(self._fields, self))
        config['dtype'] = self.dtype.name
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        return cls(**kwargs)