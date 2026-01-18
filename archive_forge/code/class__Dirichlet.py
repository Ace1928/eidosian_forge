import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
class _Dirichlet(Function):

    @staticmethod
    def forward(ctx, concentration):
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)