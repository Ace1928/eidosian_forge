import os
def _simplify_matrix(M):
    """Return a simplified copy of the matrix M"""
    assert isinstance(M, (Matrix, ImmutableMatrix))
    Mnew = M.as_mutable()
    Mnew.simplify()
    if isinstance(M, ImmutableMatrix):
        Mnew = Mnew.as_immutable()
    return Mnew