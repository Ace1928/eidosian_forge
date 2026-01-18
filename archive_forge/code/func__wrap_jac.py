import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def _wrap_jac(self, t, y, *jac_args):
    jac = self.cjac(*(t, y[::2] + 1j * y[1::2]) + jac_args)
    jac_tmp = zeros((2 * jac.shape[0], 2 * jac.shape[1]))
    jac_tmp[1::2, 1::2] = jac_tmp[::2, ::2] = real(jac)
    jac_tmp[1::2, ::2] = imag(jac)
    jac_tmp[::2, 1::2] = -jac_tmp[1::2, ::2]
    ml = getattr(self._integrator, 'ml', None)
    mu = getattr(self._integrator, 'mu', None)
    if ml is not None or mu is not None:
        jac_tmp = _transform_banded_jac(jac_tmp)
    return jac_tmp