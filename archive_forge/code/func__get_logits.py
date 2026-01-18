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
def _get_logits():
    builder = _LazyBuilder(features)
    output_tensors = []
    ordered_columns = []
    for column in sorted(feature_columns, key=lambda x: x.name):
        ordered_columns.append(column)
        with variable_scope.variable_scope(None, default_name=column._var_scope_name):
            tensor = column._get_dense_tensor(builder, weight_collections=weight_collections, trainable=trainable)
            num_elements = column._variable_shape.num_elements()
            batch_size = array_ops.shape(tensor)[0]
            output_tensor = array_ops.reshape(tensor, shape=(batch_size, num_elements))
            output_tensors.append(output_tensor)
            if cols_to_vars is not None:
                cols_to_vars[column] = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope.get_variable_scope().name)
            if cols_to_output_tensors is not None:
                cols_to_output_tensors[column] = output_tensor
    _verify_static_batch_size_equality(output_tensors, ordered_columns)
    return array_ops.concat(output_tensors, 1)