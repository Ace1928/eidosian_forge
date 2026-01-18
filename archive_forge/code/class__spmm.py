import torch
from .utils import _csr_to_coo, _transpose_with_info
class _spmm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, row_indices, values, row_offsets, column_indices, m, _transp_info):
        b = b.contiguous()
        out = torch.ops.xformers.spmm_sputnik(b, row_indices, values, row_offsets, column_indices, m)
        ctx.save_for_backward(b, row_indices, values, row_offsets, column_indices, *_transp_info)
        return out

    @staticmethod
    def backward(ctx, grad):
        b, row_indices, values, row_offsets, column_indices, *_transp_info = ctx.saved_tensors
        k = b.shape[1]
        grad = grad.contiguous()
        grad_sparse = _sddmm_func(grad, b, row_indices, row_offsets, column_indices)
        row_indices_t, values_t, row_offsets_t, column_indices_t = _transpose_with_info(values, _transp_info)
        grad_dense = torch.ops.xformers.spmm_sputnik(grad, row_indices_t, values_t, row_offsets_t, column_indices_t, k)
        return (grad_dense, None, grad_sparse, None, None, None, None)