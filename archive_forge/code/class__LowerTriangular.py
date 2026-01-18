import torch
class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """
    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]