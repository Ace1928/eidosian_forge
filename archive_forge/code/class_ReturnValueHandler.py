import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
class ReturnValueHandler:
    """
    When ltc_sync_multi is called on multi tensors, the compiled graph
    will contain output only for unique tensors - if a tensor appears multiple
    times in the input to _ltc_sync_multi, only the first occurance matters.

    However from python level, we still expect multi tensors returned with duplciation
    even if the TS graph dedup the output. e.g. for method:

      def forward(self, a):
        return a, a

    the TS graph captured by LTC will return a single tensor, but Python method expects 2.

    This class dedup the lazy tensors first to get the index that will be used
    to duplicate the eager tensors later.
    """

    def __init__(self, lazy_out_list):
        self.index: List[List[int]] = []
        self.total_count = len(lazy_out_list)
        tensor_id_to_idx: Dict[int, int] = {}
        for dup_idx, lazy_tensor in enumerate(lazy_out_list):
            uniq_idx = tensor_id_to_idx.get(id(lazy_tensor), None)
            if uniq_idx is not None:
                self.index[uniq_idx].append(dup_idx)
            else:
                uniq_idx = len(self.index)
                self.index.append([dup_idx])
                tensor_id_to_idx[id(lazy_tensor)] = uniq_idx

    def duplicate_eager_tensors(self, eager_tensor_list):
        duplicated_list = [None] * self.total_count
        assert len(eager_tensor_list) == len(self.index)
        for uniq_idx, eager_tensor in enumerate(eager_tensor_list):
            for dup_idx in self.index[uniq_idx]:
                duplicated_list[dup_idx] = eager_tensor
        return duplicated_list