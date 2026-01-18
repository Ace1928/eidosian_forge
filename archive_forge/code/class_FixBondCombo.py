from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixBondCombo(FixInternalsBase):
    """Constraint subobject for fixing linear combination of bond lengths
        within FixInternals.

        sum_i( coef_i * bond_length_i ) = constant
        """

    def prepare_jacobian(self, pos):
        bondvectors = [pos[k] - pos[h] for h, k in self.indices]
        derivs = get_distances_derivatives(bondvectors, cell=self.cell, pbc=self.pbc)
        self.finalize_jacobian(pos, len(bondvectors), 2, derivs)

    def adjust_positions(self, oldpos, newpos):
        bondvectors = [newpos[k] - newpos[h] for h, k in self.indices]
        (_,), (dists,) = conditional_find_mic([bondvectors], cell=self.cell, pbc=self.pbc)
        value = np.dot(self.coefs, dists)
        self.sigma = value - self.targetvalue
        self.finalize_positions(newpos)

    def __repr__(self):
        return 'FixBondCombo({}, {}, {})'.format(repr(self.targetvalue), self.indices, self.coefs)