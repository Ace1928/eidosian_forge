from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
class ViewNonContiguousNestedFromBuffer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor, max_seqlen: int, min_seqlen: int):
        return NestedTensor(values.detach(), offsets=offsets, lengths=lengths, _max_seqlen=max_seqlen, _min_seqlen=min_seqlen)

    @staticmethod
    def backward(ctx, gO: NestedTensor):
        return (gO.values(), None, None, None, None)