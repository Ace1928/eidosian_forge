import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
def _get_hessian(self, atoms, dx):
    """Returns the Hessian matrix d2E/dxi/dxj using a first-order
        central difference scheme with displacements dx.
        """
    N = len(atoms)
    pos = atoms.get_positions()
    hessian = np.zeros((3 * N, 3 * N))
    for i in range(3 * N):
        row = np.zeros(3 * N)
        for direction in [-1, 1]:
            disp = np.zeros(3)
            disp[i % 3] = direction * dx
            pos_disp = np.copy(pos)
            pos_disp[i // 3] += disp
            atoms.set_positions(pos_disp)
            f = atoms.get_forces()
            row += -1 * direction * f.flatten()
        row /= 2.0 * dx
        hessian[i] = row
    hessian += np.copy(hessian).T
    hessian *= 0.5
    atoms.set_positions(pos)
    return hessian