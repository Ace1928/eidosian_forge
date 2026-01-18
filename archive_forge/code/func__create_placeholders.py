import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _create_placeholders(args, kwargs, arg_names=None):
    """Create placeholders given positional args and keyword args."""
    signature_context = trace_type.InternalTracingContext(is_legacy_signature=True)
    arg_trace_types = trace_type.from_value(tuple(args), signature_context)
    kwarg_trace_types = trace_type.from_value(kwargs, signature_context)
    placeholder_mapping = signature_context.get_placeholder_mapping()
    placeholder_context = trace_type.InternalPlaceholderContext(ops.get_default_graph(), placeholder_mapping)
    if arg_names is None:
        arg_names = [None] * len(arg_trace_types.components)
    func_args = []
    for name, trace_type_arg in zip(arg_names, arg_trace_types.components):
        placeholder_context.update_naming_scope(name)
        placeholder = trace_type_arg.placeholder_value(placeholder_context)
        func_args.append(placeholder)
    func_kwargs = {}
    for name, trace_type_kwarg in zip(*sorted(kwarg_trace_types.mapping.items())):
        placeholder_context.update_naming_scope(name)
        placeholder = trace_type_kwarg.placeholder_value(placeholder_context)
        func_kwargs[name] = placeholder
    return (tuple(func_args), func_kwargs)