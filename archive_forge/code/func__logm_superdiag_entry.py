import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _logm_superdiag_entry(l1, l2, t12):
    """
    Compute a superdiagonal entry of a matrix logarithm.

    This is like Eq. (11.28) in [1]_, except the determination of whether
    l1 and l2 are sufficiently far apart has been modified.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the matrix logarithm.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    """
    if l1 == l2:
        f12 = t12 / l1
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * (np.log(l2) - np.log(l1)) / (l2 - l1)
    else:
        z = (l2 - l1) / (l2 + l1)
        u = _unwindk(np.log(l2) - np.log(l1))
        if u:
            f12 = t12 * 2 * (np.arctanh(z) + np.pi * 1j * u) / (l2 - l1)
        else:
            f12 = t12 * 2 * np.arctanh(z) / (l2 - l1)
    return f12