import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def _get_variable_specs(args):
    """Returns `VariableSpecs` from `args`."""
    variable_specs = []
    for arg in nest.flatten(args):
        if not isinstance(arg, type_spec.TypeSpec):
            continue
        if isinstance(arg, resource_variable_ops.VariableSpec):
            variable_specs.append(arg)
        elif not isinstance(arg, tensor.TensorSpec):
            variable_specs.extend(_get_variable_specs(arg._component_specs))
    return variable_specs