import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self, frame):

        def pack_hook(x):
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                gid = int(uuid.uuid4())
            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
                try:
                    with _recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)
            _internal_assert(gid in holder.handles)
            if holder.handles[gid] is None:
                raise CheckpointError('torch.utils.checkpoint: Unpack is being triggered for a tensor that was already unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do so only once. Otherwise please open an issue with details on your use case.')
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret
        if frame.unpack_error_cb is not None:

            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)