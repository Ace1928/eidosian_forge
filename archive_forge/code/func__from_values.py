import warnings
import torch
from torch.overrides import get_default_nowrap_functions
@staticmethod
def _from_values(data, mask):
    """ Differentiable constructor for MaskedTensor """

    class Constructor(torch.autograd.Function):

        @staticmethod
        def forward(ctx, data, mask):
            return MaskedTensor(data, mask)

        @staticmethod
        def backward(ctx, grad_output):
            return (grad_output, None)
    result = Constructor.apply(data, mask)
    return result