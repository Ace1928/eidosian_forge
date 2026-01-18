import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class zvode(vode):
    runner = getattr(_vode, 'zvode', None)
    supports_run_relax = 1
    supports_step = 1
    scalar = complex
    active_global_handle = 0

    def reset(self, n, has_jac):
        mf = self._determine_mf_and_set_bands(has_jac)
        if mf in (10,):
            lzw = 15 * n
        elif mf in (11, 12):
            lzw = 15 * n + 2 * n ** 2
        elif mf in (-11, -12):
            lzw = 15 * n + n ** 2
        elif mf in (13,):
            lzw = 16 * n
        elif mf in (14, 15):
            lzw = 17 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-14, -15):
            lzw = 16 * n + (2 * self.ml + self.mu) * n
        elif mf in (20,):
            lzw = 8 * n
        elif mf in (21, 22):
            lzw = 8 * n + 2 * n ** 2
        elif mf in (-21, -22):
            lzw = 8 * n + n ** 2
        elif mf in (23,):
            lzw = 9 * n
        elif mf in (24, 25):
            lzw = 10 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-24, -25):
            lzw = 9 * n + (2 * self.ml + self.mu) * n
        lrw = 20 + n
        if mf % 10 in (0, 3):
            liw = 30
        else:
            liw = 30 + n
        zwork = zeros((lzw,), complex)
        self.zwork = zwork
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, 1, 1, self.zwork, self.rwork, self.iwork, mf]
        self.success = 1
        self.initialized = False