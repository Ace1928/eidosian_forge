import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class SciPyFminPowell(SciPyGradientlessOptimizer):
    """Powell's (modified) level set method

    Uses only function calls.

    XXX: This is still a work in progress
    """

    def __init__(self, *args, **kwargs):
        """Parameters:

        direc: float
            How much to change x to initially. Defaults to 0.04.
        """
        direc = kwargs.pop('direc', None)
        SciPyGradientlessOptimizer.__init__(self, *args, **kwargs)
        if direc is None:
            self.direc = np.eye(len(self.x0()), dtype=float) * 0.04
        else:
            self.direc = np.eye(len(self.x0()), dtype=float) * direc

    def call_fmin(self, xtol, ftol, steps):
        opt.fmin_powell(self.f, self.x0(), xtol=xtol, ftol=ftol, maxiter=steps, disp=0, callback=self.callback, direc=self.direc)