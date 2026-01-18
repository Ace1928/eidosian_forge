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