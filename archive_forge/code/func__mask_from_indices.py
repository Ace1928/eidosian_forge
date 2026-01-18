import collections
from math import sin, pi, sqrt
from numbers import Real, Integral
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.utils import jsonable, lazymethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection
@staticmethod
def _mask_from_indices(atoms: Atoms, indices: Union[None, Sequence[int], np.ndarray]) -> np.ndarray:
    """Boolean mask of atoms selected by indices"""
    natoms = len(atoms)
    indices = np.asarray(indices) % natoms
    mask = np.full(natoms, False, dtype=bool)
    mask[indices] = True
    return mask