import contextlib
import dataclasses
import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
def _create_concrete_function(function_type, type_context, func_graph, tracing_options):
    """Create a `ConcreteFunction` from `args`, `kwargs`, and `func_graph`."""
    placeholder_context = trace_type.InternalPlaceholderContext(func_graph, type_context.get_placeholder_mapping())
    with func_graph.as_default():
        placeholder_bound_args = function_type.placeholder_arguments(placeholder_context)
    traced_func_graph = func_graph_module.func_graph_from_py_func(tracing_options.name, tracing_options.python_function, placeholder_bound_args.args, placeholder_bound_args.kwargs, None, func_graph=func_graph, arg_names=function_type_utils.to_arg_names(function_type), create_placeholders=False)
    transform.apply_func_graph_transforms(traced_func_graph)
    graph_capture_container = traced_func_graph.function_captures
    if tracing_options.function_captures:
        tracing_options.function_captures.merge_by_ref_with(graph_capture_container)
    output_type = trace_type.from_value(traced_func_graph.structured_outputs, type_context)
    traced_func_type = function_type_lib.FunctionType(function_type.parameters.values(), traced_func_graph.function_captures.capture_types, return_annotation=output_type)
    concrete_function = concrete_function_lib.ConcreteFunction.from_func_graph(traced_func_graph, traced_func_type, tracing_options.attributes, shared_func_graph=False)
    transform.call_concrete_function_callbacks(concrete_function)
    return concrete_function