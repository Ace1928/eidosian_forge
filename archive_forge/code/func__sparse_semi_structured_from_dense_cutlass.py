import torch
def _sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor')
    m, k = dense.shape
    device = dense.device
    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f'Invalid datatype {dense.dtype} of dense matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError('Invalid number of elements per meta element calculated')
    if m % 32 != 0:
        raise RuntimeError(f'Number rows columns of dense matrix {m} must be divisible by 32')
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(f'Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}')
    meta_ncols = k // (4 * quadbits_per_meta_elem)
    dense_4 = dense.view(-1, k // 4, 4)
    m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | bit1.to(torch.int64) << 1
    idxs1 = bit2 | bit3.to(torch.int64) << 1
    sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
    sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
    sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    meta_4 = idxs0 | idxs1 << 2
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)
    if quadbits_per_meta_elem == 4:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12
    elif quadbits_per_meta_elem == 8:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12 | meta_n[:, :, 4] << 16 | meta_n[:, :, 5] << 20 | meta_n[:, :, 6] << 24 | meta_n[:, :, 7] << 28
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = k // 2
        magic4 = [0, k // 4, 1, k // 4 + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 2 * k
        magic4 = [0, k // 2, 1, k // 2 + 1, k, 3 * k // 2, k + 1, 3 * k // 2 + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (tmp0.view(meta_ncols // 2, -1) + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)).view(-1, magic1)
    tmp2 = (torch.arange(0, 8, device=device).view(-1, 1) * torch.ones((magic0,), dtype=torch.int64, device=device) * meta_ncols).view(-1).repeat(m * meta_ncols // magic1).view(-1, magic1)
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(meta_ncols // 2, magic1)
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4
    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(m, meta_ncols)
    return (sparse, meta_reordered)