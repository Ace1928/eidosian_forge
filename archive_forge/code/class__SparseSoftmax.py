import torch
from .utils import _csr_to_coo, _transpose_with_info
class _SparseSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, n, row_indices, values, row_offsets, column_indices):
        out = torch.ops.xformers.sparse_softmax_sputnik(m, n, row_indices, values, row_offsets, column_indices)
        ctx.save_for_backward(row_indices, out, row_offsets, column_indices)
        ctx.size = (m, n)
        return out

    @staticmethod
    def backward(ctx, grad):
        row_indices, out, row_offsets, column_indices = ctx.saved_tensors
        m, n = ctx.size
        grad = grad.contiguous()
        ga = torch.ops.xformers.sparse_softmax_backward_sputnik(m, n, row_indices, out, grad, row_offsets, column_indices)
        return (None, None, None, ga, None, None)