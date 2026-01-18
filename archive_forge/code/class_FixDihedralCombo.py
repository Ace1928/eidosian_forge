from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixDihedralCombo(FixInternalsBase):
    """Constraint subobject for fixing linear combination of dihedrals
        within FixInternals.

        sum_i( coef_i * dihedral_i ) = constant
        """

    def gather_vectors(self, pos):
        v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
        v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
        v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
        return (v0, v1, v2)

    def prepare_jacobian(self, pos):
        v0, v1, v2 = self.gather_vectors(pos)
        derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell, pbc=self.pbc)
        self.finalize_jacobian(pos, len(v0), 4, derivs)

    def adjust_positions(self, oldpos, newpos):
        v0, v1, v2 = self.gather_vectors(newpos)
        value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
        value = np.dot(self.coefs, value)
        self.sigma = value - self.targetvalue
        self.finalize_positions(newpos)

    def __repr__(self):
        return 'FixDihedralCombo({}, {}, {})'.format(self.targetvalue, self.indices, self.coefs)