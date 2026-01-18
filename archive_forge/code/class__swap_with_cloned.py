import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
class _swap_with_cloned(saved_tensors_hooks):

    def __init__(self, ctx):

        def pack_hook(t):
            tid = _get_tid(t)
            sid = _get_sid(t)
            handle: Optional[_Handle] = None
            ctx.sid_to_tid[sid].add(tid)
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = t
            else:
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(tup):
            handle = tup
            error_msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' contextin which the graph was originally recorded."
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res
        super().__init__(pack_hook, unpack_hook)