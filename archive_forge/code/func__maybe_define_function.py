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
def _maybe_define_function(args, kwargs, tracing_options):
    """Gets a function for these inputs, defining it if necessary.

  Args:
    args: The varargs for the Python function.
    kwargs: The keyword args for the Python function.
    tracing_options: TracingOptions for the tracing process.

  Returns:
    A ConcreteFunction generated based on args, kwargs and tracing_options.

  Raises:
    ValueError: If inputs are incompatible with the input signature.
    TypeError: If the function inputs include non-hashable objects
    RuntimeError: If there's an internal bug (inconsistency) in handling
      shape relaxation retracing.
  """
    bound_args = function_type_utils.canonicalize_function_inputs(args, kwargs, tracing_options.polymorphic_type, tracing_options.default_values, tracing_options.is_pure)
    args, kwargs = (bound_args.args, bound_args.kwargs)
    if tracing_options.input_signature is not None:
        args = (*tracing_options.input_signature, *args[len(tracing_options.input_signature):])
    current_func_context = function_context.make_function_context(tracing_options.scope_type)
    capture_types = tracing_options.function_captures.capture_types if tracing_options.function_captures else {}
    lookup_func_type, lookup_func_context = function_type_utils.make_canonicalized_monomorphic_type(args, kwargs, capture_types, tracing_options.polymorphic_type)
    if tracing_options.function_cache is not None:
        concrete_function = tracing_options.function_cache.lookup(lookup_func_type, current_func_context)
    else:
        concrete_function = None
    if concrete_function is not None:
        return concrete_function
    with monitoring.MonitoredTimer(_graph_building_time_counter.get_cell()) if not ops.inside_function() else contextlib.nullcontext():
        with trace.Trace('tf.function-graph_building'):
            logging.vlog(1, 'Creating new FuncGraph for Python function %r (key: %r, %r)', tracing_options.python_function, current_func_context, lookup_func_type)
            logging.vlog(2, 'Python function signature [args: %s] [kwargs: %s]', args, kwargs)
            ag_status = ag_ctx.Status.ENABLED if tracing_options.autograph else ag_ctx.Status.DISABLED
            with ag_ctx.ControlStatusCtx(status=ag_status, options=tracing_options.autograph_options):
                func_graph = func_graph_module.FuncGraph(tracing_options.name)
                if tracing_options.input_signature is None and tracing_options.reduce_retracing and tracing_options.function_cache:
                    target_func_type = tracing_options.function_cache.generalize(current_func_context, lookup_func_type)
                else:
                    target_func_type = lookup_func_type
                concrete_function = _create_concrete_function(target_func_type, lookup_func_context, func_graph, tracing_options)
                if tracing_options.function_cache is not None:
                    tracing_options.function_cache.add(concrete_function, current_func_context)
                return concrete_function