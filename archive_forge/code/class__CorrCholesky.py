import torch
class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """
    event_dim = 2

    def check(self, value):
        tol = torch.finfo(value.dtype).eps * value.size(-1) * 10
        row_norm = torch.linalg.norm(value.detach(), dim=-1)
        unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim=-1)
        return _LowerCholesky().check(value) & unit_row_norm