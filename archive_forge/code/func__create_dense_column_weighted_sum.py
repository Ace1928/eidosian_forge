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
def _create_dense_column_weighted_sum(column, builder, units, weight_collections, trainable, weight_var=None):
    """Create a weighted sum of a dense column for linear_model."""
    tensor = column._get_dense_tensor(builder, weight_collections=weight_collections, trainable=trainable)
    num_elements = column._variable_shape.num_elements()
    batch_size = array_ops.shape(tensor)[0]
    tensor = array_ops.reshape(tensor, shape=(batch_size, num_elements))
    if weight_var is not None:
        weight = weight_var
    else:
        weight = variable_scope.get_variable(name='weights', shape=[num_elements, units], initializer=init_ops.zeros_initializer(), trainable=trainable, collections=weight_collections)
    return math_ops.matmul(tensor, weight, name='weighted_sum')