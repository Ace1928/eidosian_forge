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
def _sum_g_i(self) -> float:
    """Sum of the stoichiometrically weighted chemical potentials of the elements
        at specified temperature, as acquired from "g_els.json".

        Returns:
            float: sum of weighted chemical potentials [eV]
        """
    elems = self.composition.get_el_amt_dict()
    if self.interpolated:
        sum_g_i = 0
        for elem, amt in elems.items():
            g_interp = interp1d([float(t) for t in G_ELEMS], [g_dict[elem] for g_dict in G_ELEMS.values()])
            sum_g_i += amt * g_interp(self.temp)
    else:
        sum_g_i = sum((amt * G_ELEMS[str(self.temp)][elem] for elem, amt in elems.items()))
    return sum_g_i