import torch
def _round_nnz(mask, divisible_by=4):
    nonzero = torch.where(mask)
    nnz = nonzero[0].shape[0]
    nonzero = tuple((n[:nnz - nnz % divisible_by] for n in nonzero))
    nm = torch.zeros_like(mask)
    nm[nonzero] = True
    return nm