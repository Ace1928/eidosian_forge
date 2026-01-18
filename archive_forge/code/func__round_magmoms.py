from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
@no_type_check
@staticmethod
def _round_magmoms(magmoms: ArrayLike, round_magmoms_mode: float) -> np.ndarray:
    """If round_magmoms_mode is an integer, simply round to that number
        of decimal places, else if set to a float will try and round
        intelligently by grouping magmoms.
        """
    if isinstance(round_magmoms_mode, int):
        magmoms = np.around(magmoms, decimals=round_magmoms_mode)
    elif isinstance(round_magmoms_mode, float):
        try:
            range_m = max([max(magmoms), abs(min(magmoms))]) * 1.5
            kernel = gaussian_kde(magmoms, bw_method=round_magmoms_mode)
            x_grid = np.linspace(-range_m, range_m, int(1000 * range_m / round_magmoms_mode))
            kernel_m = kernel.evaluate(x_grid)
            extrema = x_grid[argrelextrema(kernel_m, comparator=np.greater)]
            magmoms = [extrema[np.abs(extrema - m).argmin()] for m in magmoms]
        except Exception as exc:
            warnings.warn('Failed to round magmoms intelligently, falling back to simple rounding.')
            warnings.warn(str(exc))
        n_decimals = len(str(round_magmoms_mode).split('.')[1]) + 1
        magmoms = np.around(magmoms, decimals=n_decimals)
    return np.array(magmoms)