import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
@contextlib.contextmanager
def _coalescing_manager(group: Optional[ProcessGroup]=None, device: Optional[torch.device]=None, async_ops: Optional[bool]=False):
    """
    Context manager used to coalesce collectives or P2P operations when possible.

    Args:
        group (`ProcessGroup`, optional): The process group to work on. If None,
            the default process group will be used.
        device (`torch.device`, optional): Default is None, set to a device if
            there isn't a `**_coalesced` implementation by the backend.
        async_ops (`bool`, optional): whether the coalesced ops are async ops.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # Synchronous ops
        >>> with _coalescing_manager():
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> # Asynchronous ops
        >>> with _coalescing_manager(async_ops=True) as cm:
        >>>     for i in range(num_colls):
        >>>         dist.all_reduce(tensors[i])
        >>> cm.wait()

    .. warning::
       :func:`_coalescing_manager` currently do not support coalescing
       all-reduces with different reduce operators, e.g.  `ReduceOp.SUM` mixed
       with `ReduceOp.PRODUCT`.
    """
    group = group or _get_default_group()
    op_list = _world.pg_coalesce_state.setdefault(group, [])
    if op_list:
        raise ValueError('ProcessGroup has non-empty op list at the start of coalescing')
    if device:
        group._start_coalescing(device)
    cm = _CoalescingManager()
    yield cm
    op_list = _world.pg_coalesce_state.pop(group)
    if op_list:
        op0 = op_list[0].op
        if op0 == all_reduce:
            tensors = []
            for op in op_list:
                tensors.append(op.tensor)
            opts = AllreduceCoalescedOptions()
            opts.reduceOp = op_list[0].redop
            work = group.allreduce_coalesced(tensors, opts)
        elif op0 == all_gather_into_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(op.dst_tensor)
            work = group.allgather_into_tensor_coalesced(outputs, inputs)
        elif op0 == reduce_scatter_tensor:
            inputs = []
            outputs = []
            for op in op_list:
                inputs.append(op.tensor)
                outputs.append(op.dst_tensor)
                opts = ReduceScatterOptions()
                opts.reduceOp = op_list[0].redop
            work = group.reduce_scatter_tensor_coalesced(outputs, inputs, opts)
        else:
            raise AssertionError(f'Coalescing manager does not support fast-path coalescing of {op0}, yet {op0} is still recorded in op list. This is an internal error of c10d.')
    if device:
        work = group._end_coalescing(device)
    if async_ops:
        cm.append(work)
    else:
        work.wait()