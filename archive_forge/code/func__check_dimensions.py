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
def _check_dimensions(atoms: Atoms, hessian: np.ndarray, indices: Sequence[int], two_d: bool=False) -> int:
    """Sanity check on array shapes from input data

        Args:
            atoms: Structure
            indices: Indices of atoms used in Hessian
            hessian: Proposed Hessian array

        Returns:
            Number of atoms contributing to Hessian

        Raises:
            ValueError if Hessian dimensions are not (N, 3, N, 3)

        """
    n_atoms = len(atoms[indices])
    if two_d:
        ref_shape = [n_atoms * 3, n_atoms * 3]
        ref_shape_txt = '{n:d}x{n:d}'.format(n=n_atoms * 3)
    else:
        ref_shape = [n_atoms, 3, n_atoms, 3]
        ref_shape_txt = '{n:d}x3x{n:d}x3'.format(n=n_atoms)
    if isinstance(hessian, np.ndarray) and hessian.shape == tuple(ref_shape):
        return n_atoms
    else:
        raise ValueError('Hessian for these atoms should be a {} numpy array.'.format(ref_shape_txt))