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
@contextmanager
def _use_context(parent_context: '_opentelemetry.Context') -> Generator[None, None, None]:
    """Uses the Ray trace context for the span."""
    if parent_context is not None:
        new_context = parent_context
    else:
        new_context = _opentelemetry.Context()
    token = _opentelemetry.context.attach(new_context)
    try:
        yield
    finally:
        _opentelemetry.context.detach(token)