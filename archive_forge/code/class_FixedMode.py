from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixedMode(FixConstraint):
    """Constrain atoms to move along directions orthogonal to
    a given mode only."""

    def __init__(self, mode):
        self.mode = (np.asarray(mode) / np.sqrt((mode ** 2).sum())).reshape(-1)

    def get_removed_dof(self, atoms):
        return len(atoms)

    def adjust_positions(self, atoms, newpositions):
        newpositions = newpositions.ravel()
        oldpositions = atoms.positions.ravel()
        step = newpositions - oldpositions
        newpositions -= self.mode * np.dot(step, self.mode)

    def adjust_forces(self, atoms, forces):
        forces = forces.ravel()
        forces -= self.mode * np.dot(forces, self.mode)

    def index_shuffle(self, atoms, ind):
        eps = 1e-12
        mode = self.mode.reshape(-1, 3)
        excluded = np.ones(len(mode), dtype=bool)
        excluded[ind] = False
        if (abs(mode[excluded]) > eps).any():
            raise IndexError('All nonzero parts of mode not in slice')
        self.mode = mode[ind].ravel()

    def get_indices(self):
        return []

    def todict(self):
        return {'name': 'FixedMode', 'kwargs': {'mode': self.mode.tolist()}}

    def __repr__(self):
        return 'FixedMode(%s)' % self.mode.tolist()