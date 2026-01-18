from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class TargetScoreFunction(torch.autograd.Function):
    """Custom checkpointed function to compute the target score."""

    @staticmethod
    def get_target_score(i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, full_precision: bool, margin: float, scale: Optional[float]) -> torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        if scale is not None:
            target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
        else:
            target_score = i * tw
        if full_precision:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)
        if scale is not None:
            target_score -= margin
            target_score *= scale
        return target_score

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput') -> torch.Tensor:
        """Forward, without activations."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG target fwd')
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(i, w, target, kernel_obj.fp_target, kernel_obj.margin, kernel_obj.scale)
        return x

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Forward and backward again, assign or accumulate the gradients."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG target bwd')
        assert len(args) == 1
        i, w, target = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert not target.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            scores = TargetScoreFunction.get_target_score(i, w, target, ctx.kernel_obj.fp_target, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return (i.grad, None, None, None)