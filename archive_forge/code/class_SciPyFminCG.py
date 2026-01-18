import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class SciPyFminCG(SciPyOptimizer):
    """Non-linear (Polak-Ribiere) conjugate gradient algorithm"""

    def call_fmin(self, fmax, steps):
        output = opt.fmin_cg(self.f, self.x0(), fprime=self.fprime, gtol=fmax * 0.1, norm=np.inf, maxiter=steps, full_output=1, disp=0, callback=self.callback)
        warnflag = output[-1]
        if warnflag == 2:
            raise OptimizerConvergenceError('Warning: Desired error not necessarily achieved due to precision loss')