import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@dataclass
class SubclassCreationMeta:
    """
    Used for AOTDispatch.
    This dataclass gives us the information we need to reconstruct a tensor subclass
    from our flat inputs.
    Why is this important? The graph that we'd like to trace out contains flat tensor inputs,
    But the user's original model may have subclass inputs and outputs.
    So we need to wrap/unwrap subclasses as necessary to translate between the user's
    view (subclass inps/outs), and the backend compiler's view (graph with no subclass args).

    Complications arise mostly from the fact that a subclass can hold more than one inner tensor;
    So for a given subclass input/output, we need to carefully track which indices map
    to the subclass tensor in the corresponding "dense-tensor-only" graph.
    """
    flat_tensor_start_idx: int
    arg_count: int
    original_subclass: torch.Tensor
    meta: Any
    inner_keys: List[Any]

    def creation_fn(self, all_args, *, is_runtime: bool):
        curr_args = all_args[self.flat_tensor_start_idx:self.flat_tensor_start_idx + self.arg_count]
        assert len(curr_args) == len(self.inner_keys), f'inner_keys: {str(self.inner_keys)}. len(curr_args): {len(curr_args)}'
        out = type(self.original_subclass).__tensor_unflatten__(dict(zip(self.inner_keys, curr_args)), self.meta)
        if not is_runtime:
            torch._mirror_autograd_meta_to(self.original_subclass, out)
        return out

    def __post_init__(self):
        assert is_fake(self.original_subclass)