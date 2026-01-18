import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized
class FakeStructuredSparsity(nn.Module):
    """
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        assert isinstance(self.mask, torch.Tensor)
        assert self.mask.shape[0] == x.shape[0]
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    def state_dict(self, *args, **kwargs):
        return {}