from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _integrate_gsl(self, *args, **kwargs):
    """ Do not use directly (use ``integrate(..., integrator='gsl')``).

        Uses `GNU Scientific Library <http://www.gnu.org/software/gsl/>`_
        (via `pygslodeiv2 <https://pypi.python.org/pypi/pygslodeiv2>`_)
        to integrate the ODE system.

        Parameters
        ----------
        \\*args :
            see :meth:`integrate`
        method : str (default: 'bsimp')
            what stepper to use, see :py:attr:`gslodeiv2.steppers`
        \\*\\*kwargs :
            keyword arguments passed onto
            :py:func:`gslodeiv2.integrate_adaptive`/:py:func:`gslodeiv2.integrate_predefined`

        Returns
        -------
        See :meth:`integrate`
        """
    import pygslodeiv2
    kwargs['with_jacobian'] = kwargs.get('method', 'bsimp') in pygslodeiv2.requires_jac
    return self._integrate(pygslodeiv2.integrate_adaptive, pygslodeiv2.integrate_predefined, *args, **kwargs)