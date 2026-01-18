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
def _inject_tracing_into_class(_cls):
    """Given a class that will be made into an actor,
    inject tracing into all of the methods."""

    def span_wrapper(method: Callable[..., Any]) -> Any:

        def _resume_span(self: Any, *_args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **_kwargs: Any) -> Any:
            """
            Wrap the user's function with a function that
            will extract the trace context
            """
            if not _is_tracing_enabled() or _ray_trace_ctx is None:
                return method(self, *_args, **_kwargs)
            tracer: _opentelemetry.trace.Tracer = _opentelemetry.trace.get_tracer(__name__)
            with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_actor_span_consumer_name(self.__class__.__name__, method), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_actor_hydrate_span_args(self.__class__.__name__, method)):
                return method(self, *_args, **_kwargs)
        return _resume_span

    def async_span_wrapper(method: Callable[..., Any]) -> Any:

        async def _resume_span(self: Any, *_args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **_kwargs: Any) -> Any:
            """
            Wrap the user's function with a function that
            will extract the trace context
            """
            if not _is_tracing_enabled() or _ray_trace_ctx is None:
                return await method(self, *_args, **_kwargs)
            tracer = _opentelemetry.trace.get_tracer(__name__)
            with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_actor_span_consumer_name(self.__class__.__name__, method.__name__), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_actor_hydrate_span_args(self.__class__.__name__, method.__name__)):
                return await method(self, *_args, **_kwargs)
        return _resume_span
    methods = inspect.getmembers(_cls, is_function_or_method)
    for name, method in methods:
        if is_static_method(_cls, name) or is_class_method(method):
            continue
        if inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method):
            continue
        if name == '__del__':
            continue
        setattr(method, '__signature__', _add_param_to_signature(method, inspect.Parameter('_ray_trace_ctx', inspect.Parameter.KEYWORD_ONLY, default=None)))
        if inspect.iscoroutinefunction(method):
            wrapped_method = wraps(method)(async_span_wrapper(method))
        else:
            wrapped_method = wraps(method)(span_wrapper(method))
        setattr(_cls, name, wrapped_method)
    return _cls