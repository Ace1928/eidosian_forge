from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_scipy(self, intern_x0, tol=1e-08, method=None, **kwargs):
    """ Uses ``scipy.optimize.root``

        See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

        Parameters
        ----------
        intern_x0: array_like
            initial guess
        tol: float
            Tolerance
        method: str
            What method to use. Defaults to ``'lm'`` if ``self.nf > self.nx`` otherwise ``'hybr'``.

        """
    from scipy.optimize import root
    if method is None:
        if self.nf > self.nx:
            method = 'lm'
        elif self.nf == self.nx:
            method = 'hybr'
        else:
            raise ValueError('Underdetermined problem')
    if 'band' in kwargs:
        raise ValueError("Set 'band' at initialization instead.")
    if 'args' in kwargs:
        raise ValueError("Set 'args' as params in initialization instead.")
    new_kwargs = kwargs.copy()
    if self.band is not None:
        warnings.warn('Band argument ignored (see SciPy docs)')
        new_kwargs['band'] = self.band
    new_kwargs['args'] = self.internal_params
    return root(self.f_cb, intern_x0, jac=self.j_cb, method=method, tol=tol, **new_kwargs)