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
class _CrossedColumn(_CategoricalColumn, collections.namedtuple('_CrossedColumn', ['keys', 'hash_bucket_size', 'hash_key'])):
    """See `crossed_column`."""

    @property
    def name(self):
        feature_names = []
        for key in _collect_leaf_level_keys(self):
            if isinstance(key, _FeatureColumn):
                feature_names.append(key.name)
            else:
                feature_names.append(key)
        return '_X_'.join(sorted(feature_names))

    @property
    def _parse_example_spec(self):
        config = {}
        for key in self.keys:
            if isinstance(key, _FeatureColumn):
                config.update(key._parse_example_spec)
            else:
                config.update({key: parsing_ops.VarLenFeature(dtypes.string)})
        return config

    def _transform_feature(self, inputs):
        feature_tensors = []
        for key in _collect_leaf_level_keys(self):
            if isinstance(key, six.string_types):
                feature_tensors.append(inputs.get(key))
            elif isinstance(key, _CategoricalColumn):
                ids_and_weights = key._get_sparse_tensors(inputs)
                if ids_and_weights.weight_tensor is not None:
                    raise ValueError('crossed_column does not support weight_tensor, but the given column populates weight_tensor. Given column: {}'.format(key.name))
                feature_tensors.append(ids_and_weights.id_tensor)
            else:
                raise ValueError('Unsupported column type. Given: {}'.format(key))
        return sparse_ops.sparse_cross_hashed(inputs=feature_tensors, num_buckets=self.hash_bucket_size, hash_key=self.hash_key)

    @property
    def _num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.hash_bucket_size

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        return _CategoricalColumn.IdWeightPair(inputs.get(self), None)