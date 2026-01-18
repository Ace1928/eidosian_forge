from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def _get_basis_spglib(atoms: Atoms, tol: float=1e-05) -> np.ndarray:
    """Get a reduced basis using spglib. This requires having the
    spglib package installed.

    :param atoms: Atoms, atoms object to get basis from
    :param tol: ``float``, numeric tolerance for positional comparisons
        Default: ``1e-5``
    """
    if not _has_spglib():
        raise ImportError('This function requires spglib. Use "get_basis" and specify the spacegroup instead, or install spglib.')
    scaled_positions = atoms.get_scaled_positions()
    reduced_indices = _get_reduced_indices(atoms, tol=tol)
    return scaled_positions[reduced_indices]