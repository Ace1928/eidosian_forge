from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def _hermite_normal_form(A):
    """
    Compute the Hermite Normal Form of DomainMatrix *A* over :ref:`ZZ`.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over domain :ref:`ZZ`.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.5.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    m, n = A.shape
    A = A.to_dense().rep.copy()
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            break
        k -= 1
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = (A[i][k] // d, A[i][j] // d)
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        if b == 0:
            k += 1
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    return DomainMatrix.from_rep(A)[:, k:]