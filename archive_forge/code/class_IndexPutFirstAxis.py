import torch
import torch.nn.functional as F
from einops import rearrange, repeat
class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_values = grad_output[indices]
        return (grad_values, None, None)