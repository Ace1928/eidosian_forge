import torch
def _dense3d_to_sparse(matrix, device):
    assert len(matrix.shape) == 3
    mask = matrix != 0
    if not torch.all(mask == mask[0]):
        raise ValueError('Expected the same sparsity pattern over the batch dimension')
    mask = _round_nnz(mask[0], divisible_by=4)
    mask = mask[None].expand(matrix.shape)
    values = matrix[mask].reshape(matrix.shape[0], -1).to(device)
    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(mask[0], device)
    return (values, row_indices, row_offsets, column_indices)