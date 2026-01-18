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
class SharedEmbeddingColumn(DenseColumn, SequenceDenseColumn, fc_old._DenseColumn, fc_old._SequenceDenseColumn, collections.namedtuple('SharedEmbeddingColumn', ('categorical_column', 'shared_embedding_column_creator', 'combiner', 'max_norm', 'use_safe_embedding_lookup'))):
    """See `embedding_column`."""

    def __new__(cls, categorical_column, shared_embedding_column_creator, combiner, max_norm, use_safe_embedding_lookup=True):
        return super(SharedEmbeddingColumn, cls).__new__(cls, categorical_column=categorical_column, shared_embedding_column_creator=shared_embedding_column_creator, combiner=combiner, max_norm=max_norm, use_safe_embedding_lookup=use_safe_embedding_lookup)

    @property
    def _is_v2_column(self):
        return True

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_shared_embedding'.format(self.categorical_column.name)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    def _parse_example_spec(self):
        return _raise_shared_embedding_column_error()

    def transform_feature(self, transformation_cache, state_manager):
        """See `FeatureColumn` base class."""
        return transformation_cache.get(self.categorical_column, state_manager)

    def _transform_feature(self, inputs):
        return _raise_shared_embedding_column_error()

    @property
    def variable_shape(self):
        """See `DenseColumn` base class."""
        return tensor_shape.TensorShape([self.shared_embedding_column_creator.dimension])

    @property
    def _variable_shape(self):
        return _raise_shared_embedding_column_error()

    def _get_dense_tensor_internal(self, transformation_cache, state_manager):
        """Private method that follows the signature of _get_dense_tensor."""
        with ops.name_scope(None, default_name=self.name):
            sparse_tensors = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
            sparse_ids = sparse_tensors.id_tensor
            sparse_weights = sparse_tensors.weight_tensor
            embedding_weights = self.shared_embedding_column_creator.embedding_weights
            sparse_id_rank = tensor_shape.dimension_value(sparse_ids.dense_shape.get_shape()[0])
            embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
            if not self.use_safe_embedding_lookup and sparse_id_rank is not None and (sparse_id_rank <= 2):
                embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
            return embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights, combiner=self.combiner, name='%s_weights' % self.name, max_norm=self.max_norm)

    def get_dense_tensor(self, transformation_cache, state_manager):
        """Returns the embedding lookup result."""
        if isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must not be of type SequenceCategoricalColumn. Suggested fix A: If you wish to use DenseFeatures, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use SequenceFeatures instead of DenseFeatures. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        return self._get_dense_tensor_internal(transformation_cache, state_manager)

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        return _raise_shared_embedding_column_error()

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        """See `SequenceDenseColumn` base class."""
        if not isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must be of type SequenceCategoricalColumn to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        dense_tensor = self._get_dense_tensor_internal(transformation_cache, state_manager)
        sparse_tensors = self.categorical_column.get_sparse_tensors(transformation_cache, state_manager)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        return _raise_shared_embedding_column_error()

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.categorical_column]