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
def _initial_guess(self):
    """
        Quadratic fit to get an initial guess for the parameters.

        Returns:
            tuple: 4 floats for (e0, b0, b1, v0)
        """
    a, b, c = np.polyfit(self.volumes, self.energies, 2)
    self.eos_params = [a, b, c]
    v0 = -b / (2 * a)
    e0 = a * v0 ** 2 + b * v0 + c
    b0 = 2 * a * v0
    b1 = 4
    vol_min, vol_max = (min(self.volumes), max(self.volumes))
    if not vol_min < v0 and v0 < vol_max:
        raise EOSError('The minimum volume of a fitted parabola is not in the input volumes.')
    return (e0, b0, b1, v0)