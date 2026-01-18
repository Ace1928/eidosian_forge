import abc
import contextlib
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _convert_variables_to_tensors(self):
    """Recursively converts ResourceVariables in the LinearOperator to Tensors.

    The usage of `self._type_spec._from_components` violates the contract of
    `CompositeTensor`, since it is called on a different nested structure
    (one containing only `Tensor`s) than `self.type_spec` specifies (one that
    may contain `ResourceVariable`s). Since `LinearOperator`'s
    `_from_components` method just passes the contents of the nested structure
    to `__init__` to rebuild the operator, and any `LinearOperator` that may be
    instantiated with `ResourceVariables` may also be instantiated with
    `Tensor`s, this usage is valid.

    Returns:
      tensor_operator: `self` with all internal Variables converted to Tensors.
    """
    components = self._type_spec._to_components(self)
    tensor_components = variable_utils.convert_variables_to_tensors(components)
    return self._type_spec._from_components(tensor_components)