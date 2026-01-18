import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def _solout(self, nr, xold, x, y, nd, icomp, con):
    if self.solout is not None:
        if self.solout_cmplx:
            y = y[::2] + 1j * y[1::2]
        return self.solout(x, y)
    else:
        return 1