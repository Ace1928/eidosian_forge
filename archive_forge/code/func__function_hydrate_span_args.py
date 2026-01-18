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
def _function_hydrate_span_args(function_name: str):
    """Get the Attributes of the function that will be reported as attributes
    in the trace."""
    runtime_context = get_runtime_context()
    span_args = {'ray.remote': 'function', 'ray.function': function_name, 'ray.pid': str(os.getpid()), 'ray.job_id': runtime_context.get_job_id(), 'ray.node_id': runtime_context.get_node_id()}
    if ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE:
        task_id = runtime_context.get_task_id()
        if task_id:
            span_args['ray.task_id'] = task_id
    worker_id = getattr(ray._private.worker.global_worker, 'worker_id', None)
    if worker_id:
        span_args['ray.worker_id'] = worker_id.hex()
    return span_args