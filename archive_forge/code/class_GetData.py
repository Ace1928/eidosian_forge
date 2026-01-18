import warnings
import torch
from torch.overrides import get_default_nowrap_functions
class GetData(torch.autograd.Function):

    @staticmethod
    def forward(ctx, self):
        return self._masked_data

    @staticmethod
    def backward(ctx, grad_output):
        if is_masked_tensor(grad_output):
            return grad_output
        return MaskedTensor(grad_output, self.get_mask())