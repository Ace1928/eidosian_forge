import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
class TagFilter:
    """Filter which constrains same-tag atoms to behave
    like internally rigid moieties.
    """

    def __init__(self, atoms):
        self.atoms = atoms
        gather_atoms_by_tag(self.atoms)
        self.tags = self.atoms.get_tags()
        self.unique_tags = np.unique(self.tags)
        self.n = len(self.unique_tags)

    def get_positions(self):
        all_pos = self.atoms.get_positions()
        cop_pos = np.zeros((self.n, 3))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            cop_pos[i] = np.average(all_pos[indices], axis=0)
        return cop_pos

    def set_positions(self, positions, **kwargs):
        cop_pos = self.get_positions()
        all_pos = self.atoms.get_positions()
        assert np.all(np.shape(positions) == np.shape(cop_pos))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            shift = positions[i] - cop_pos[i]
            all_pos[indices] += shift
        self.atoms.set_positions(all_pos, **kwargs)

    def get_forces(self, *args, **kwargs):
        f = self.atoms.get_forces()
        forces = np.zeros((self.n, 3))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            forces[i] = np.sum(f[indices], axis=0)
        return forces

    def get_masses(self):
        m = self.atoms.get_masses()
        masses = np.zeros(self.n)
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            masses[i] = np.sum(m[indices])
        return masses

    def __len__(self):
        return self.n