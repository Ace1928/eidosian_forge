import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
class ProfileOperatorsTorchDispatchMode(TorchDispatchMode):

    def __init__(self, num_runs: int=10) -> None:
        self.data: List[ProfileMetadata] = []
        self.num_runs: int = num_runs

    def _get_inplace_metadata(self, func, out) -> Tuple[int, int, Tuple[int, ...]]:
        curr_idx = len(self.data)

        def get_tensor_id(e):
            return e.untyped_storage().data_ptr() if isinstance(e, torch.Tensor) else None
        output_ids = tree_map(get_tensor_id, out)
        if not is_inplace(func):
            return (curr_idx, output_ids, ())
        op_id = curr_idx
        op_parent_id = -1
        for i, d in enumerate(self.data):
            past_output_ids = d.output_ids
            past_output_ids = [past_output_ids] if not isinstance(past_output_ids, (list, tuple, dict)) else past_output_ids
            if output_ids in past_output_ids:
                op_parent_id = i
                break
        if op_parent_id < 0:
            op_parent_id = op_id
        inplace_info = (op_id, op_parent_id)
        return (curr_idx, output_ids, inplace_info)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        curr_idx, output_ids, inplace_info = self._get_inplace_metadata(func, out)
        is_view_like = is_view_fn(func) or is_inplace_view_fn(func)
        is_rand_op = torch.Tag.nondeterministic_seeded in func.tags
        if func.overloadpacket.__name__ == '_scaled_dot_product_flash_attention':
            is_rand_op = kwargs.get('dropout_p', 0) != 0
        torch.cuda.synchronize()
        t = time.time()
        for i in range(self.num_runs):
            func(*args, **kwargs)
        torch.cuda.synchronize()
        time_taken = (time.time() - t) / self.num_runs
        torch.cuda.reset_peak_memory_stats()
        mem1 = torch.cuda.max_memory_allocated() / 2 ** 20
        func(*args, **kwargs)
        mem2 = torch.cuda.max_memory_allocated() / 2 ** 20
        self.data.append(ProfileMetadata(func, time_taken, mem2 - mem1, curr_idx, output_ids, inplace_info, is_view_like, is_rand_op))
        return out