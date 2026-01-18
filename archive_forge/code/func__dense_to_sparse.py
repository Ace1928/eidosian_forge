import torch
def _dense_to_sparse(matrix, device):
    """Converts dense 2d matrix to a csr sparse matrix."""
    assert len(matrix.shape) == 2
    value_dtype = torch.float32
    mask = matrix != 0
    values = matrix[mask].to(dtype=value_dtype, device=device)
    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(mask, device)
    return (values, row_indices, row_offsets, column_indices)