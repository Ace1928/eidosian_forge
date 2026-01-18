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
class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self, target_frame_ref: ReferenceType, gid: int):

        def pack_hook(x):
            target_frame = target_frame_ref()
            assert target_frame is not None
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1
            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    target_frame.ignore_saved_mismatch = True
                    return x.detach()
                raise CheckpointError('torch.utils.checkpoint: trying to save more tensors during recomputation than during the original forward pass.')
            holder = target_frame.weak_holders[recomp_idx]()
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x.detach()
            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(target_frame.weak_holders):
                raise _StopRecomputationError()
            return x.detach()

        def unpack_hook(x):
            return x
        super().__init__(pack_hook, unpack_hook)