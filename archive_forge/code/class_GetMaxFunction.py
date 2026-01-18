from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class GetMaxFunction(torch.autograd.Function):
    """Custom checkpointed function to get max-per-token from an input and a weight"""

    @staticmethod
    def get_max(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, w_idx: int, full_precision: bool, margin: float, scale: Optional[float]) -> torch.Tensor:
        """
        Throughout this code:

          i: input data with shape = (split-of-tokens, d_model)
          w: weight data with shape = (split-of-vocabs, d_model)
          tgt: target prediction data with shape = (split-of-tokens,)
        """
        if scale is not None:
            _m = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _m = torch.matmul(i, w.T)
        if full_precision:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput', w_idx: int, w_split_size: int, split_dim: int) -> torch.Tensor:
        """Forward function that computes the max, without saving activations."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG max fwd')
        ctx.save_for_backward(i, w, tgt)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, tgt, w_idx, kernel_obj.fp_max, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Recompute the forward max and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG max bwd')
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            maxs = GetMaxFunction.get_max(i, w, tgt, ctx.w_idx, ctx.kernel_obj.fp_max, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(maxs, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return (i.grad, None, None, None, None, None, None)