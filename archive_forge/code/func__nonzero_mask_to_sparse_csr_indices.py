import torch
def _nonzero_mask_to_sparse_csr_indices(mask, device):
    """Converts dense 2d matrix to a csr sparse matrix."""
    assert len(mask.shape) == 2
    index_dtype = torch.int32
    row_offsets = mask.sum(dim=-1, dtype=index_dtype).cumsum(dim=-1, dtype=index_dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))
    row_indices = _diffsort(row_offsets).to(index_dtype)
    column_indices = torch.where(mask)[1].to(index_dtype).contiguous()
    row_indices = row_indices.to(device)
    row_offsets = row_offsets.to(device)
    column_indices = column_indices.to(device)
    return (row_indices, row_offsets, column_indices)