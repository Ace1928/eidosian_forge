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
class _DenseColumn(_FeatureColumn):
    """Represents a column which can be represented as `Tensor`.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  Some examples of this type are: numeric_column, embedding_column,
  indicator_column.
  """

    @abc.abstractproperty
    def _variable_shape(self):
        """`TensorShape` of `_get_dense_tensor`, without batch dimension."""
        pass

    @abc.abstractmethod
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        """Returns a `Tensor`.

    The output of this function will be used by model-builder-functions. For
    example the pseudo code of `input_layer` will be like:

    ```python
    def input_layer(features, feature_columns, ...):
      outputs = [fc._get_dense_tensor(...) for fc in feature_columns]
      return tf.concat(outputs)
    ```

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: List of graph collections to which Variables (if any
        will be created) are added.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).

    Returns:
      `Tensor` of shape [batch_size] + `_variable_shape`.
    """
        pass