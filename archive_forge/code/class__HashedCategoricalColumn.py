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
class _HashedCategoricalColumn(_CategoricalColumn, collections.namedtuple('_HashedCategoricalColumn', ['key', 'hash_bucket_size', 'dtype'])):
    """see `categorical_column_with_hash_bucket`."""

    @property
    def name(self):
        return self.key

    @property
    def _parse_example_spec(self):
        return {self.key: parsing_ops.VarLenFeature(self.dtype)}

    def _transform_feature(self, inputs):
        input_tensor = _to_sparse_input_and_drop_ignore_values(inputs.get(self.key))
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

    @property
    def _num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.hash_bucket_size

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        return _CategoricalColumn.IdWeightPair(inputs.get(self), None)