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
class _LinearOperatorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """Composite tensor gradient for `LinearOperator`."""

    def get_gradient_components(self, value):
        return value._type_spec._to_components(value)

    def replace_gradient_components(self, value, components):
        flat_components = nest.flatten(components)
        if all((c is None for c in flat_components)):
            return None
        value_components = value._type_spec._to_components(value)
        flat_grad_components = []
        for gc, vc in zip(flat_components, nest.flatten(value_components)):
            if gc is None:
                flat_grad_components.append(nest.map_structure(lambda x: array_ops.zeros_like(x, dtype=value.dtype), vc, expand_composites=True))
            else:
                flat_grad_components.append(gc)
        grad_components = nest.pack_sequence_as(value_components, flat_grad_components)
        return value._type_spec._from_components(grad_components)