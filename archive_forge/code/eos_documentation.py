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

        Fit energies as function of volumes.

        Args:
            volumes (list/np.array)
            energies (list/np.array)

        Returns:
            EOSBase: EOSBase object
        