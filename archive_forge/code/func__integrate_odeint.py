from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _integrate_odeint(self, *args, **kwargs):
    """ Do not use directly (use ``integrate(..., integrator='odeint')``).

        Uses `Boost.Numeric.Odeint <http://www.odeint.com>`_
        (via `pyodeint <https://pypi.python.org/pypi/pyodeint>`_) to integrate
        the ODE system.
        """
    import pyodeint
    kwargs['with_jacobian'] = kwargs.get('method', 'rosenbrock4') in pyodeint.requires_jac
    return self._integrate(pyodeint.integrate_adaptive, pyodeint.integrate_predefined, *args, **kwargs)