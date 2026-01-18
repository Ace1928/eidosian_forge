import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class dop853(dopri5):
    runner = getattr(_dop, 'dop853', None)
    name = 'dop853'

    def __init__(self, rtol=1e-06, atol=1e-12, nsteps=500, max_step=0.0, first_step=0.0, safety=0.9, ifactor=6.0, dfactor=0.3, beta=0.0, method=None, verbosity=-1):
        super().__init__(rtol, atol, nsteps, max_step, first_step, safety, ifactor, dfactor, beta, method, verbosity)

    def reset(self, n, has_jac):
        work = zeros((11 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, self._solout, self.iout, self.work, self.iwork]
        self.success = 1