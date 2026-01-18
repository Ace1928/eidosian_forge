import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
    """Performs the vibrational analysis."""
    hessian = self._get_hessian(atoms, dx)
    if massweighing:
        m = np.array([np.repeat(atoms.get_masses() ** (-0.5), 3)])
        hessian *= m * m.T
    eigvals, eigvecs = np.linalg.eigh(hessian)
    modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
    return modes