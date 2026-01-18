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
class _BucketizedColumn(_DenseColumn, _CategoricalColumn, collections.namedtuple('_BucketizedColumn', ['source_column', 'boundaries'])):
    """See `bucketized_column`."""

    @property
    def name(self):
        return '{}_bucketized'.format(self.source_column.name)

    @property
    def _parse_example_spec(self):
        return self.source_column._parse_example_spec

    def _transform_feature(self, inputs):
        source_tensor = inputs.get(self.source_column)
        return math_ops._bucketize(source_tensor, boundaries=self.boundaries)

    @property
    def _variable_shape(self):
        return tensor_shape.TensorShape(tuple(self.source_column.shape) + (len(self.boundaries) + 1,))

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        input_tensor = inputs.get(self)
        return array_ops.one_hot(indices=math_ops.cast(input_tensor, dtypes.int64), depth=len(self.boundaries) + 1, on_value=1.0, off_value=0.0)

    @property
    def _num_buckets(self):
        return (len(self.boundaries) + 1) * self.source_column.shape[0]

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        """Converts dense inputs to SparseTensor so downstream code can use it."""
        input_tensor = inputs.get(self)
        batch_size = array_ops.shape(input_tensor)[0]
        source_dimension = self.source_column.shape[0]
        i1 = array_ops.reshape(array_ops.tile(array_ops.expand_dims(math_ops.range(0, batch_size), 1), [1, source_dimension]), (-1,))
        i2 = array_ops.tile(math_ops.range(0, source_dimension), [batch_size])
        bucket_indices = array_ops.reshape(input_tensor, (-1,)) + (len(self.boundaries) + 1) * i2
        indices = math_ops.cast(array_ops.transpose(array_ops_stack.stack((i1, i2))), dtypes.int64)
        dense_shape = math_ops.cast(array_ops_stack.stack([batch_size, source_dimension]), dtypes.int64)
        sparse_tensor = sparse_tensor_lib.SparseTensor(indices=indices, values=bucket_indices, dense_shape=dense_shape)
        return _CategoricalColumn.IdWeightPair(sparse_tensor, None)