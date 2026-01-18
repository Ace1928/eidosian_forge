import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
class _CloneArgBeforeMutateMode(TorchDispatchMode):

    def __init__(self, ctx):
        self.ctx = ctx

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                t = kwargs['out'] if arg.is_out else args[idx]
                tid = _get_tid(t)
                sid = _get_sid(t)
                ctx = self.ctx
                if sid in ctx.sid_to_tid:
                    for tid in ctx.sid_to_tid[sid]:
                        if tid not in ctx.tid_to_weakhandle:
                            continue
                        handle = ctx.tid_to_weakhandle[tid]
                        if handle in ctx.cloned:
                            continue
                        ctx.cloned[handle] = ctx.original[handle].clone()
                        del ctx.original[handle]
        rs = func(*args, **kwargs)
        return rs