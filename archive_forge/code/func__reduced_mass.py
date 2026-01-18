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
def _reduced_mass(structure) -> float:
    """Reduced mass as calculated via Eq. 6 in Bartel et al. (2018).

        Args:
            structure (Structure): The pymatgen Structure object of the entry.

        Returns:
            float: reduced mass (amu)
        """
    reduced_comp = structure.composition.reduced_composition
    n_elems = len(reduced_comp.elements)
    elem_dict = reduced_comp.get_el_amt_dict()
    denominator = (n_elems - 1) * reduced_comp.num_atoms
    all_pairs = combinations(elem_dict.items(), 2)
    mass_sum = 0
    for pair in all_pairs:
        m_i = Composition(pair[0][0]).weight
        m_j = Composition(pair[1][0]).weight
        alpha_i = pair[0][1]
        alpha_j = pair[1][1]
        mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j)
    return 1 / denominator * mass_sum