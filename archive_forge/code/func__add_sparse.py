import cupy
from cupyx import cusparse
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
def _add_sparse(self, other, alpha, beta):
    self.sum_duplicates()
    other = other.tocsc().T
    other.sum_duplicates()
    if cusparse.check_availability('csrgeam2'):
        csrgeam = cusparse.csrgeam2
    elif cusparse.check_availability('csrgeam'):
        csrgeam = cusparse.csrgeam
    else:
        raise NotImplementedError
    return csrgeam(self.T, other, alpha, beta).T