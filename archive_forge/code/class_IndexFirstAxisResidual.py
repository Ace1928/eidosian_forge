import torch
import torch.nn.functional as F
from einops import rearrange, repeat
class IndexFirstAxisResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = (input.shape[0], input.shape[1:])
        second_dim = other_shape.numel()
        output = input[indices]
        return (output, input.detach())

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        indices = indices.reshape(indices.shape[0], *(1,) * (grad_output.ndim - 1))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return (grad_input.reshape(ctx.first_axis_dim, *other_shape), None)