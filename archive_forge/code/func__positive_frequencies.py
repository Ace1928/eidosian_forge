from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
@lazy_property
def _positive_frequencies(self) -> np.ndarray:
    """Numpy array containing the list of positive frequencies."""
    return self.frequencies[self.ind_zero_freq:]