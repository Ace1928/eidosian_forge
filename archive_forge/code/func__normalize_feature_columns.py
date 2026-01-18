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
def _normalize_feature_columns(feature_columns):
    """Normalizes the `feature_columns` input.

  This method converts the `feature_columns` to list type as best as it can. In
  addition, verifies the type and other parts of feature_columns, required by
  downstream library.

  Args:
    feature_columns: The raw feature columns, usually passed by users.

  Returns:
    The normalized feature column list.

  Raises:
    ValueError: for any invalid inputs, such as empty, duplicated names, etc.
  """
    if isinstance(feature_columns, _FeatureColumn):
        feature_columns = [feature_columns]
    if isinstance(feature_columns, collections_abc.Iterator):
        feature_columns = list(feature_columns)
    if isinstance(feature_columns, dict):
        raise ValueError('Expected feature_columns to be iterable, found dict.')
    for column in feature_columns:
        if not isinstance(column, _FeatureColumn):
            raise ValueError('Items of feature_columns must be a _FeatureColumn. Given (type {}): {}.'.format(type(column), column))
    if not feature_columns:
        raise ValueError('feature_columns must not be empty.')
    name_to_column = {}
    for column in feature_columns:
        if column.name in name_to_column:
            raise ValueError('Duplicate feature column name found for columns: {} and {}. This usually means that these columns refer to same base feature. Either one must be discarded or a duplicated but renamed item must be inserted in features dict.'.format(column, name_to_column[column.name]))
        name_to_column[column.name] = column
    return feature_columns