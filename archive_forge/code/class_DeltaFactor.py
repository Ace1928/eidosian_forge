from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import leastsq, minimize
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
class DeltaFactor(PolynomialEOS):
    """Fitting a polynomial EOS using delta factor."""

    def _func(self, volume, params):
        x = volume ** (-2 / 3.0)
        return np.poly1d(list(params))(x)

    def fit(self, order=3):
        """Overridden since this eos works with volume**(2/3) instead of volume."""
        x = self.volumes ** (-2 / 3.0)
        self.eos_params = np.polyfit(x, self.energies, order)
        self._set_params()

    def _set_params(self):
        """
        Overridden to account for the fact the fit with volume**(2/3) instead
        of volume.
        """
        deriv0 = np.poly1d(self.eos_params)
        deriv1 = np.polyder(deriv0, 1)
        deriv2 = np.polyder(deriv1, 1)
        deriv3 = np.polyder(deriv2, 1)
        for x in np.roots(deriv1):
            if x > 0 and deriv2(x) > 0:
                v0 = x ** (-3 / 2.0)
                break
        else:
            raise EOSError('No minimum could be found')
        derivV2 = 4 / 9 * x ** 5 * deriv2(x)
        derivV3 = -20 / 9 * x ** (13 / 2.0) * deriv2(x) - 8 / 27 * x ** (15 / 2.0) * deriv3(x)
        b0 = derivV2 / x ** (3 / 2.0)
        b1 = -1 - x ** (-3 / 2.0) * derivV3 / derivV2
        self._params = [deriv0(v0 ** (-2 / 3.0)), b0, b1, v0]