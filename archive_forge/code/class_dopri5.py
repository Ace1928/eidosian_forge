import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class dopri5(IntegratorBase):
    runner = getattr(_dop, 'dopri5', None)
    name = 'dopri5'
    supports_solout = True
    messages = {1: 'computation successful', 2: 'computation successful (interrupted by solout)', -1: 'input is not consistent', -2: 'larger nsteps is needed', -3: 'step size becomes too small', -4: 'problem is probably stiff (interrupted)'}

    def __init__(self, rtol=1e-06, atol=1e-12, nsteps=500, max_step=0.0, first_step=0.0, safety=0.9, ifactor=10.0, dfactor=0.2, beta=0.0, method=None, verbosity=-1):
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.max_step = max_step
        self.first_step = first_step
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        self.beta = beta
        self.verbosity = verbosity
        self.success = 1
        self.set_solout(None)

    def set_solout(self, solout, complex=False):
        self.solout = solout
        self.solout_cmplx = complex
        if solout is None:
            self.iout = 0
        else:
            self.iout = 1

    def reset(self, n, has_jac):
        work = zeros((8 * n + 21,), float)
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

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        x, y, iwork, istate = self.runner(*(f, t0, y0, t1) + tuple(self.call_args) + (f_params,))
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__, self.messages.get(istate, unexpected_istate_msg)), stacklevel=2)
            self.success = 0
        return (y, x)

    def _solout(self, nr, xold, x, y, nd, icomp, con):
        if self.solout is not None:
            if self.solout_cmplx:
                y = y[::2] + 1j * y[1::2]
            return self.solout(x, y)
        else:
            return 1