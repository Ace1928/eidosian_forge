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
def _enable_tracing():
    global _global_is_tracing_enabled, _opentelemetry
    _global_is_tracing_enabled = True
    _opentelemetry = _OpenTelemetryProxy()
    _opentelemetry.try_all()