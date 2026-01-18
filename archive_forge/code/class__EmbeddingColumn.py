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
class _EmbeddingColumn(_DenseColumn, _SequenceDenseColumn, collections.namedtuple('_EmbeddingColumn', ('categorical_column', 'dimension', 'combiner', 'layer_creator', 'ckpt_to_load_from', 'tensor_name_in_ckpt', 'max_norm', 'trainable', 'use_safe_embedding_lookup'))):
    """See `embedding_column`."""

    def __new__(cls, categorical_column, dimension, combiner, layer_creator, ckpt_to_load_from, tensor_name_in_ckpt, max_norm, trainable, use_safe_embedding_lookup=True):
        return super(_EmbeddingColumn, cls).__new__(cls, categorical_column=categorical_column, dimension=dimension, combiner=combiner, layer_creator=layer_creator, ckpt_to_load_from=ckpt_to_load_from, tensor_name_in_ckpt=tensor_name_in_ckpt, max_norm=max_norm, trainable=trainable, use_safe_embedding_lookup=use_safe_embedding_lookup)

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = '{}_embedding'.format(self.categorical_column.name)
        return self._name

    @property
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec

    def _transform_feature(self, inputs):
        return inputs.get(self.categorical_column)

    @property
    def _variable_shape(self):
        if not hasattr(self, '_shape'):
            self._shape = tensor_shape.TensorShape([self.dimension])
        return self._shape

    def _get_dense_tensor_internal(self, inputs, weight_collections=None, trainable=None):
        """Private method that follows the signature of _get_dense_tensor."""
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs, weight_collections=weight_collections, trainable=trainable)
        sparse_ids = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor
        embedding_weights = self.layer_creator(weight_collections=weight_collections, scope=variable_scope.get_variable_scope())
        if self.ckpt_to_load_from is not None:
            to_restore = embedding_weights
            if isinstance(to_restore, variables.PartitionedVariable):
                to_restore = to_restore._get_variable_list()
            checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {self.tensor_name_in_ckpt: to_restore})
        sparse_id_rank = tensor_shape.dimension_value(sparse_ids.dense_shape.get_shape()[0])
        embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
        if not self.use_safe_embedding_lookup and sparse_id_rank is not None and (sparse_id_rank <= 2):
            embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
        return embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights, combiner=self.combiner, name='%s_weights' % self.name, max_norm=self.max_norm)

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if isinstance(self.categorical_column, _SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must not be of type _SequenceCategoricalColumn. Suggested fix A: If you wish to use input_layer, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use sequence_input_layer instead of input_layer. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        return self._get_dense_tensor_internal(inputs=inputs, weight_collections=weight_collections, trainable=trainable)

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if not isinstance(self.categorical_column, _SequenceCategoricalColumn):
            raise ValueError('In embedding_column: {}. categorical_column must be of type _SequenceCategoricalColumn to use sequence_input_layer. Suggested fix: Use one of sequence_categorical_column_with_*. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        dense_tensor = self._get_dense_tensor_internal(inputs=inputs, weight_collections=weight_collections, trainable=trainable)
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)
        return _SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)