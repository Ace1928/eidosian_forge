from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
class _MaskedToSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError('MaskedToSparse forward: input must be a MaskedTensor.')
        if input.layout == torch.sparse_coo:
            return input
        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_coo().coalesce()
        sparse_data = data.sparse_mask(sparse_mask)
        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()