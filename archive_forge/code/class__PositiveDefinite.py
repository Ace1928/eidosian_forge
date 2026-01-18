import torch
class _PositiveDefinite(_Symmetric):
    """
    Constrain to positive-definite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.cholesky_ex(value).info.eq(0)