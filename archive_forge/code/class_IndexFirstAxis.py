import torch
import torch.nn.functional as F
from einops import rearrange, repeat
class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = (input.shape[0], input.shape[1:])
        second_dim = other_shape.numel()
        return torch.gather(rearrange(input, 'b ... -> b (...)'), 0, repeat(indices, 'z -> z d', d=second_dim)).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype)
        grad_input.scatter_(0, repeat(indices, 'z -> z d', d=grad_output.shape[1]), grad_output)
        return (grad_input.reshape(ctx.first_axis_dim, *other_shape), None)