import importlib
import inspect
import logging
import os
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter
from types import ModuleType
from typing import (
import ray
import ray._private.worker
from ray._private.inspect_util import (
from ray.runtime_context import get_runtime_context
def _inject_tracing_into_function(function):
    """Wrap the function argument passed to RemoteFunction's __init__ so that
    future execution of that function will include tracing.
    Use the provided trace context from kwargs.
    """
    if not _is_tracing_enabled():
        return function
    setattr(function, '__signature__', _add_param_to_signature(function, inspect.Parameter('_ray_trace_ctx', inspect.Parameter.KEYWORD_ONLY, default=None)))

    @wraps(function)
    def _function_with_tracing(*args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        if _ray_trace_ctx is None:
            return function(*args, **kwargs)
        tracer = _opentelemetry.trace.get_tracer(__name__)
        function_name = function.__module__ + '.' + function.__name__
        with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_function_span_consumer_name(function_name), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_function_hydrate_span_args(function_name)):
            return function(*args, **kwargs)
    return _function_with_tracing