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
def _get_dense_tensor_internal(self, inputs, weight_collections=None, trainable=None):
    """Private method that follows the signature of _get_dense_tensor."""
    with ops.name_scope(None, default_name=self.name):
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs, weight_collections=weight_collections, trainable=trainable)
        sparse_ids = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor
        embedding_shape = (self.categorical_column._num_buckets, self.dimension)
        shared_embedding_collection = ops.get_collection(self.shared_embedding_collection_name)
        if shared_embedding_collection:
            if len(shared_embedding_collection) > 1:
                raise ValueError('Collection {} can only contain one variable. Suggested fix A: Choose a unique name for this collection. Suggested fix B: Do not add any variables to this collection. The feature_column library already adds a variable under the hood.'.format(shared_embedding_collection))
            embedding_weights = shared_embedding_collection[0]
            if embedding_weights.get_shape() != embedding_shape:
                raise ValueError('Shared embedding collection {} contains variable {} of unexpected shape {}. Expected shape is {}. Suggested fix A: Choose a unique name for this collection. Suggested fix B: Do not add any variables to this collection. The feature_column library already adds a variable under the hood.'.format(self.shared_embedding_collection_name, embedding_weights.name, embedding_weights.get_shape(), embedding_shape))
        else:
            embedding_weights = variable_scope.get_variable(name='embedding_weights', shape=embedding_shape, dtype=dtypes.float32, initializer=self.initializer, trainable=self.trainable and trainable, collections=weight_collections)
            ops.add_to_collection(self.shared_embedding_collection_name, embedding_weights)
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