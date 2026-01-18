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
class BucketizedColumn(DenseColumn, CategoricalColumn, fc_old._DenseColumn, fc_old._CategoricalColumn, collections.namedtuple('BucketizedColumn', ('source_column', 'boundaries'))):
    """See `bucketized_column`."""

    @property
    def _is_v2_column(self):
        return isinstance(self.source_column, fc_types.FeatureColumn) and self.source_column._is_v2_column

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_bucketized'.format(self.source_column.name)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.source_column.parse_example_spec

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.source_column._parse_example_spec

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        """Returns bucketized categorical `source_column` tensor."""
        source_tensor = inputs.get(self.source_column)
        return math_ops._bucketize(source_tensor, boundaries=self.boundaries)

    def transform_feature(self, transformation_cache, state_manager):
        """Returns bucketized categorical `source_column` tensor."""
        source_tensor = transformation_cache.get(self.source_column, state_manager)
        return math_ops._bucketize(source_tensor, boundaries=self.boundaries)

    @property
    def variable_shape(self):
        """See `DenseColumn` base class."""
        return tensor_shape.TensorShape(tuple(self.source_column.shape) + (len(self.boundaries) + 1,))

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _variable_shape(self):
        return self.variable_shape

    def _get_dense_tensor_for_input_tensor(self, input_tensor):
        return array_ops.one_hot(indices=math_ops.cast(input_tensor, dtypes.int64), depth=len(self.boundaries) + 1, on_value=1.0, off_value=0.0)

    def get_dense_tensor(self, transformation_cache, state_manager):
        """Returns one hot encoded dense `Tensor`."""
        input_tensor = transformation_cache.get(self, state_manager)
        return self._get_dense_tensor_for_input_tensor(input_tensor)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        input_tensor = inputs.get(self)
        return self._get_dense_tensor_for_input_tensor(input_tensor)

    @property
    def num_buckets(self):
        """See `CategoricalColumn` base class."""
        return (len(self.boundaries) + 1) * self.source_column.shape[0]

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _num_buckets(self):
        return self.num_buckets

    def _get_sparse_tensors_for_input_tensor(self, input_tensor):
        batch_size = array_ops.shape(input_tensor)[0]
        source_dimension = self.source_column.shape[0]
        i1 = array_ops.reshape(array_ops.tile(array_ops.expand_dims(math_ops.range(0, batch_size), 1), [1, source_dimension]), (-1,))
        i2 = array_ops.tile(math_ops.range(0, source_dimension), [batch_size])
        bucket_indices = array_ops.reshape(input_tensor, (-1,)) + (len(self.boundaries) + 1) * i2
        indices = math_ops.cast(array_ops.transpose(array_ops_stack.stack((i1, i2))), dtypes.int64)
        dense_shape = math_ops.cast(array_ops_stack.stack([batch_size, source_dimension]), dtypes.int64)
        sparse_tensor = sparse_tensor_lib.SparseTensor(indices=indices, values=bucket_indices, dense_shape=dense_shape)
        return CategoricalColumn.IdWeightPair(sparse_tensor, None)

    def get_sparse_tensors(self, transformation_cache, state_manager):
        """Converts dense inputs to SparseTensor so downstream code can use it."""
        input_tensor = transformation_cache.get(self, state_manager)
        return self._get_sparse_tensors_for_input_tensor(input_tensor)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        """Converts dense inputs to SparseTensor so downstream code can use it."""
        del weight_collections
        del trainable
        input_tensor = inputs.get(self)
        return self._get_sparse_tensors_for_input_tensor(input_tensor)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.source_column]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import serialize_feature_column
        config = dict(zip(self._fields, self))
        config['source_column'] = serialize_feature_column(self.source_column)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import deserialize_feature_column
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['source_column'] = deserialize_feature_column(config['source_column'], custom_objects, columns_by_name)
        return cls(**kwargs)