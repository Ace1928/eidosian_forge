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
def _start_span(self, args: Sequence[Any]=None, kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any) -> Any:
    if not _is_tracing_enabled() or self._actor_ref()._ray_is_cross_language:
        if kwargs is not None:
            assert '_ray_trace_ctx' not in kwargs
        return method(self, args, kwargs, *_args, **_kwargs)
    class_name = self._actor_ref()._ray_actor_creation_function_descriptor.class_name
    method_name = self._method_name
    assert '_ray_trace_ctx' not in _kwargs
    tracer = _opentelemetry.trace.get_tracer(__name__)
    with tracer.start_as_current_span(name=_actor_span_producer_name(class_name, method_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_actor_hydrate_span_args(class_name, method_name)) as span:
        kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
        span.set_attribute('ray.actor_id', self._actor_ref()._ray_actor_id.hex())
        return method(self, args, kwargs, *_args, **_kwargs)