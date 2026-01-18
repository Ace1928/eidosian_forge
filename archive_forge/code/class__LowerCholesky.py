import torch
class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """
    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        return lower_triangular & positive_diagonal