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
class _WeightedCategoricalColumn(_CategoricalColumn, collections.namedtuple('_WeightedCategoricalColumn', ('categorical_column', 'weight_feature_key', 'dtype'))):
    """See `weighted_categorical_column`."""

    @property
    def name(self):
        return '{}_weighted_by_{}'.format(self.categorical_column.name, self.weight_feature_key)

    @property
    def _parse_example_spec(self):
        config = self.categorical_column._parse_example_spec
        if self.weight_feature_key in config:
            raise ValueError('Parse config {} already exists for {}.'.format(config[self.weight_feature_key], self.weight_feature_key))
        config[self.weight_feature_key] = parsing_ops.VarLenFeature(self.dtype)
        return config

    @property
    def _num_buckets(self):
        return self.categorical_column._num_buckets

    def _transform_feature(self, inputs):
        weight_tensor = inputs.get(self.weight_feature_key)
        if weight_tensor is None:
            raise ValueError('Missing weights {}.'.format(self.weight_feature_key))
        weight_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(weight_tensor)
        if self.dtype != weight_tensor.dtype.base_dtype:
            raise ValueError('Bad dtype, expected {}, but got {}.'.format(self.dtype, weight_tensor.dtype))
        if not isinstance(weight_tensor, sparse_tensor_lib.SparseTensor):
            weight_tensor = _to_sparse_input_and_drop_ignore_values(weight_tensor, ignore_value=0.0)
        if not weight_tensor.dtype.is_floating:
            weight_tensor = math_ops.cast(weight_tensor, dtypes.float32)
        return (inputs.get(self.categorical_column), weight_tensor)

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        tensors = inputs.get(self)
        return _CategoricalColumn.IdWeightPair(tensors[0], tensors[1])