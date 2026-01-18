import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _report_nonhermitian(M, name):
    """
    Report if `M` is not a hermitian matrix given its type.
    """
    md = M - M.T.conj()
    nmd = linalg.norm(md, 1)
    tol = 10 * cupy.finfo(M.dtype).eps
    tol *= max(1, float(linalg.norm(M, 1)))
    if nmd > tol:
        warnings.warn(f'Matrix {name} of the type {M.dtype} is not Hermitian: condition: {nmd} < {tol} fails.', UserWarning, stacklevel=4)