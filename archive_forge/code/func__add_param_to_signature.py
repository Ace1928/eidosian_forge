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
def _add_param_to_signature(function: Callable, new_param: Parameter):
    """Add additional Parameter to function signature."""
    old_sig = inspect.signature(function)
    old_sig_list_repr = list(old_sig.parameters.values())
    if any((param.name == new_param.name for param in old_sig_list_repr)):
        return old_sig
    new_params = _sort_params_list(old_sig_list_repr + [new_param])
    new_sig = old_sig.replace(parameters=new_params)
    return new_sig