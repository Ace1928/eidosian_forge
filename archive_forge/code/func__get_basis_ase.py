from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def _get_basis_ase(atoms: Atoms, spacegroup: _SPACEGROUP, tol: float=1e-05) -> np.ndarray:
    """Recursively get a reduced basis, by removing equivalent sites.
    Uses the first index as a basis, then removes all equivalent sites,
    uses the next index which hasn't been placed into a basis, etc.

    :param atoms: Atoms object to get basis from.
    :param spacegroup: ``int``, ``str``, or
        :class:`ase.spacegroup.Spacegroup` object.
    :param tol: ``float``, numeric tolerance for positional comparisons
        Default: ``1e-5``
    """
    scaled_positions = atoms.get_scaled_positions()
    spacegroup = Spacegroup(spacegroup)

    def scaled_in_sites(scaled_pos: np.ndarray, sites: np.ndarray):
        """Check if a scaled position is in a site"""
        for site in sites:
            if np.allclose(site, scaled_pos, atol=tol):
                return True
        return False

    def _get_basis(scaled_positions: np.ndarray, spacegroup: Spacegroup, all_basis=None) -> np.ndarray:
        """Main recursive function to be executed"""
        if all_basis is None:
            all_basis = []
        if len(scaled_positions) == 0:
            return np.array(all_basis)
        basis = scaled_positions[0]
        all_basis.append(basis.tolist())
        sites, _ = spacegroup.equivalent_sites(basis)
        new_scaled = np.array([sc for sc in scaled_positions if not scaled_in_sites(sc, sites)])
        assert len(new_scaled) < len(scaled_positions)
        return _get_basis(new_scaled, spacegroup, all_basis=all_basis)
    return _get_basis(scaled_positions, spacegroup)