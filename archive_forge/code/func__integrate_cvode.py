from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _integrate_cvode(self, *args, **kwargs):
    """ Do not use directly (use ``integrate(..., integrator='cvode')``).

        Uses CVode from CVodes in
        `SUNDIALS <https://computation.llnl.gov/casc/sundials/>`_
        (via `pycvodes <https://pypi.python.org/pypi/pycvodes>`_)
        to integrate the ODE system. """
    import pycvodes
    kwargs['with_jacobian'] = kwargs.get('method', 'bdf') in pycvodes.requires_jac
    if 'lband' in kwargs or 'uband' in kwargs or 'band' in kwargs:
        raise ValueError('lband and uband set locally (set at initialization instead)')
    if self.band is not None:
        kwargs['lband'], kwargs['uband'] = self.band
    kwargs['autonomous_exprs'] = self.autonomous_exprs
    return self._integrate(pycvodes.integrate_adaptive, pycvodes.integrate_predefined, *args, **kwargs)