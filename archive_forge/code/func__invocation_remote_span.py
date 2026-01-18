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
@wraps(method)
def _invocation_remote_span(self, args: Any=None, kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any) -> Any:
    if not _is_tracing_enabled() or self._is_cross_language:
        if kwargs is not None:
            assert '_ray_trace_ctx' not in kwargs
        return method(self, args, kwargs, *_args, **_kwargs)
    assert '_ray_trace_ctx' not in kwargs
    tracer = _opentelemetry.trace.get_tracer(__name__)
    with tracer.start_as_current_span(_function_span_producer_name(self._function_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_function_hydrate_span_args(self._function_name)):
        kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
        return method(self, args, kwargs, *_args, **_kwargs)