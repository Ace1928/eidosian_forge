import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class IntegratorBase:
    runner = None
    success = None
    istate = None
    supports_run_relax = None
    supports_step = None
    supports_solout = False
    integrator_classes = []
    scalar = float

    def acquire_new_handle(self):
        self.__class__.active_global_handle += 1
        self.handle = self.__class__.active_global_handle

    def check_handle(self):
        if self.handle is not self.__class__.active_global_handle:
            raise IntegratorConcurrencyError(self.__class__.__name__)

    def reset(self, n, has_jac):
        """Prepare integrator for call: allocate memory, set flags, etc.
        n - number of equations.
        has_jac - if user has supplied function for evaluating Jacobian.
        """

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t=t1 using y0 as an initial condition.
        Return 2-tuple (y1,t1) where y1 is the result and t=t1
        defines the stoppage coordinate of the result.
        """
        raise NotImplementedError('all integrators must define run(f, jac, t0, t1, y0, f_params, jac_params)')

    def step(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Make one integration step and return (y1,t1)."""
        raise NotImplementedError('%s does not support step() method' % self.__class__.__name__)

    def run_relax(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t>=t1 and return (y1,t)."""
        raise NotImplementedError('%s does not support run_relax() method' % self.__class__.__name__)