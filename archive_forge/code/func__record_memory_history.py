import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def _record_memory_history(enabled='all', *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.cuda.memory._snapshot()`.

    In addition too keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.cuda.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currenly allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)