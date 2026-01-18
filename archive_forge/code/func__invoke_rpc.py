import collections
import contextlib
import functools
import inspect
import logging
import threading
from typing import Dict, Generic, TypeVar, Set, Any, TYPE_CHECKING
import torch
from torch.futures import Future
from torch._C._distributed_rpc import (
from .internal import (
from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT
from ._utils import _group_membership_management, _update_group_membership
def _invoke_rpc(to, func, rpc_type, args=None, kwargs=None, rpc_timeout: float=UNSET_RPC_TIMEOUT):
    if not callable(func):
        raise TypeError('function should be callable.')
    qualified_name = torch.jit._builtins._find_builtin(func)
    dst_worker_info = _to_worker_info(to)
    should_profile = _get_should_profile()
    ctx_manager = _enable_rpc_profiler(should_profile, qualified_name, func, rpc_type, dst_worker_info)
    with ctx_manager as rf:
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        is_async_exec = hasattr(func, '_wrapped_async_rpc_function')
        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped
        if qualified_name is not None:
            fut = _invoke_rpc_builtin(dst_worker_info, qualified_name, rpc_timeout, *args, **kwargs)
        elif isinstance(func, torch.jit.ScriptFunction):
            fut = _invoke_rpc_torchscript(dst_worker_info.name, torch._jit_internal._qualified_name(func), args, kwargs, rpc_timeout, is_async_exec)
        else:
            pickled_python_udf, tensors = _default_pickler.serialize(PythonUDF(func, args, kwargs))
            fut = _invoke_rpc_python_udf(dst_worker_info, pickled_python_udf, tensors, rpc_timeout, is_async_exec)
        if should_profile:
            assert torch.autograd._profiler_enabled()
            assert rf is not None
            fut = rf._call_end_callbacks_on_future(fut)
    return fut