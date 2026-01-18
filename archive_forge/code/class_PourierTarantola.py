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
class PourierTarantola(EOSBase):
    """PourierTarantola EOS."""

    def _func(self, volume, params):
        """Pourier-Tarantola equation from PRB 70, 224107."""
        e0, b0, b1, v0 = tuple(params)
        eta = (volume / v0) ** (1 / 3)
        squiggle = -3 * np.log(eta)
        return e0 + b0 * v0 * squiggle ** 2 / 6 * (3 + squiggle * (b1 - 2))