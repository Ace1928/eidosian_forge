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
class IndicatorColumn(DenseColumn, SequenceDenseColumn, fc_old._DenseColumn, fc_old._SequenceDenseColumn, collections.namedtuple('IndicatorColumn', 'categorical_column')):
    """Represents a one-hot column for use in deep networks.

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` function.
  """

    @property
    def _is_v2_column(self):
        return isinstance(self.categorical_column, fc_types.FeatureColumn) and self.categorical_column._is_v2_column

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_indicator'.format(self.categorical_column.name)

    def _transform_id_weight_pair(self, id_weight_pair, size):
        id_tensor = id_weight_pair.id_tensor
        weight_tensor = id_weight_pair.weight_tensor
        if weight_tensor is not None:
            weighted_column = sparse_ops.sparse_merge(sp_ids=id_tensor, sp_values=weight_tensor, vocab_size=int(size))
            weighted_column = sparse_ops.sparse_slice(weighted_column, [0, 0], weighted_column.dense_shape)
            return array_ops.scatter_nd(weighted_column.indices, weighted_column.values, weighted_column.dense_shape)
        dense_id_tensor = sparse_ops.sparse_tensor_to_dense(id_tensor, default_value=-1)
        one_hot_id_tensor = array_ops.one_hot(dense_id_tensor, depth=size, on_value=1.0, off_value=0.0)
        return math_ops.reduce_sum(one_hot_id_tensor, axis=[-2])

    def transform_feature(self, transformation_cache, state_manager):
        """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.

    Raises:
      ValueError: if input rank is not known at graph building time.
    """
        id_weight_pair = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
        return self._transform_id_weight_pair(id_weight_pair, self.variable_shape[-1])

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        id_weight_pair = self.categorical_column._get_sparse_tensors(inputs)
        return self._transform_id_weight_pair(id_weight_pair, self._variable_shape[-1])

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec

    @property
    def variable_shape(self):
        """Returns a `TensorShape` representing the shape of the dense `Tensor`."""
        if isinstance(self.categorical_column, fc_types.FeatureColumn):
            return tensor_shape.TensorShape([1, self.categorical_column.num_buckets])
        else:
            return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _variable_shape(self):
        return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])

    def get_dense_tensor(self, transformation_cache, state_manager):
        """Returns dense `Tensor` representing feature.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Dense `Tensor` created within `transform_feature`.

    Raises:
      ValueError: If `categorical_column` is a `SequenceCategoricalColumn`.
    """
        if isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In indicator_column: {}. categorical_column must not be of type SequenceCategoricalColumn. Suggested fix A: If you wish to use DenseFeatures, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use SequenceFeatures instead of DenseFeatures. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        return transformation_cache.get(self, state_manager)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        if isinstance(self.categorical_column, (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):
            raise ValueError('In indicator_column: {}. categorical_column must not be of type _SequenceCategoricalColumn. Suggested fix A: If you wish to use DenseFeatures, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use SequenceFeatures instead of DenseFeatures. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        return inputs.get(self)

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        """See `SequenceDenseColumn` base class."""
        if not isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In indicator_column: {}. categorical_column must be of type SequenceCategoricalColumn to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        dense_tensor = transformation_cache.get(self, state_manager)
        sparse_tensors = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        if not isinstance(self.categorical_column, (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):
            raise ValueError('In indicator_column: {}. categorical_column must be of type _SequenceCategoricalColumn to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        dense_tensor = inputs.get(self)
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.categorical_column]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import serialize_feature_column
        config = dict(zip(self._fields, self))
        config['categorical_column'] = serialize_feature_column(self.categorical_column)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import deserialize_feature_column
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['categorical_column'] = deserialize_feature_column(config['categorical_column'], custom_objects, columns_by_name)
        return cls(**kwargs)