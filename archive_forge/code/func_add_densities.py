from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def add_densities(density1: Mapping[Spin, ArrayLike], density2: Mapping[Spin, ArrayLike]) -> dict[Spin, np.ndarray]:
    """Sum two densities.

    Args:
        density1: First density.
        density2: Second density.

    Returns:
        dict[Spin, np.ndarray]
    """
    return {spin: np.array(density1[spin]) + np.array(density2[spin]) for spin in density1}