import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class SciPyFmin(SciPyGradientlessOptimizer):
    """Nelder-Mead Simplex algorithm

    Uses only function calls.

    XXX: This is still a work in progress
    """

    def call_fmin(self, xtol, ftol, steps):
        opt.fmin(self.f, self.x0(), xtol=xtol, ftol=ftol, maxiter=steps, disp=0, callback=self.callback)