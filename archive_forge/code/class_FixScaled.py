from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixScaled(FixConstraintSingle):
    """Fix an atom index *a* in the directions of the unit vectors."""

    def __init__(self, cell, a, mask=(1, 1, 1)):
        self.cell = np.asarray(cell)
        self.a = a
        self.mask = np.array(mask, bool)

    def get_removed_dof(self, atoms):
        return self.mask.sum()

    def adjust_positions(self, atoms, new):
        scaled_old = atoms.cell.scaled_positions(atoms.positions)
        scaled_new = atoms.cell.scaled_positions(new)
        for n in range(3):
            if self.mask[n]:
                scaled_new[self.a, n] = scaled_old[self.a, n]
        new[self.a] = atoms.cell.cartesian_positions(scaled_new)[self.a]

    def adjust_forces(self, atoms, forces):
        scaled_forces = atoms.cell.cartesian_positions(forces)
        scaled_forces[self.a] *= -(self.mask - 1)
        forces[self.a] = atoms.cell.scaled_positions(scaled_forces)[self.a]

    def todict(self):
        return {'name': 'FixScaled', 'kwargs': {'a': self.a, 'cell': self.cell.tolist(), 'mask': self.mask.tolist()}}

    def __repr__(self):
        return 'FixScaled(%s, %d, %s)' % (repr(self.cell), self.a, repr(self.mask))