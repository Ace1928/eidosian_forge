from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixDihedral(FixDihedralCombo):
    """Constraint object for fixing a dihedral angle using
        the SHAKE algorithm. This one allows also other constraints.

        SHAKE convergence is potentially problematic for near-undefined
        dihedral angles (i.e. when one of the two angles a012 or a123
        approaches 0 or 180 degrees).
        """

    def __init__(self, targetvalue, indices, masses, cell, pbc):
        indices = [list(indices) + [1.0]]
        super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

    def adjust_positions(self, oldpos, newpos):
        v0, v1, v2 = self.gather_vectors(newpos)
        value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
        self.sigma = (value - self.targetvalue + 180) % 360 - 180
        self.finalize_positions(newpos)

    def __repr__(self):
        return 'FixDihedral({}, {})'.format(self.targetvalue, *self.indices)