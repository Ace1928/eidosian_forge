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
class _CategoricalColumn(_FeatureColumn):
    """Represents a categorical feature.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  A categorical feature typically handled with a `tf.sparse.SparseTensor` of
  IDs.
  """
    IdWeightPair = collections.namedtuple('IdWeightPair', ['id_tensor', 'weight_tensor'])

    @abc.abstractproperty
    def _num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        pass

    @abc.abstractmethod
    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      inputs: A `LazyBuilder` as a cache to get input tensors required to create
        `IdWeightPair`.
      weight_collections: List of graph collections to which variables (if any
        will be created) are added.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.compat.v1.get_variable`).
    """
        pass