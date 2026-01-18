from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
@staticmethod
def _g_delta_sisso(vol_per_atom, reduced_mass, temp) -> float:
    """G^delta as predicted by SISSO-learned descriptor from Eq. (4) in
        Bartel et al. (2018).

        Args:
            vol_per_atom (float): volume per atom [Å^3/atom]
            reduced_mass (float): as calculated with pair-wise sum formula
                [amu]
            temp (float): Temperature [K]

        Returns:
            float: G^delta [eV/atom]
        """
    return (-0.000248 * np.log(vol_per_atom) - 8.94e-05 * reduced_mass / vol_per_atom) * temp + 0.181 * np.log(temp) - 0.882