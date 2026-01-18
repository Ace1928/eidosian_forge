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