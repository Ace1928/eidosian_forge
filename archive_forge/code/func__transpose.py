import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def _transpose(self):
    """ Default implementation of _transpose; defers to rmatvec + conj"""
    return _TransposedLinearOperator(self)