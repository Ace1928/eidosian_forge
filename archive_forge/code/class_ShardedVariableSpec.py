import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
class ShardedVariableSpec(type_spec.TypeSpec):
    """Type specification for a `ShardedVariable`."""
    __slots__ = ['_variable_specs']
    value_type = property(lambda self: ShardedVariable)

    def __init__(self, *variable_specs):
        self._variable_specs = tuple(variable_specs)

    def _serialize(self):
        return self._variable_specs

    @property
    def _component_specs(self):
        return self._variable_specs

    def _to_components(self, value):
        return tuple(value.variables)

    def _from_components(self, variables):
        return ShardedVariable(variables)

    def _cast(self, value, _):
        return value