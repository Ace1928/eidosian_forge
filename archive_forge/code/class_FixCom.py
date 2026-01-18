from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixCom(FixConstraint):
    """Constraint class for fixing the center of mass.

    References

    https://pubs.acs.org/doi/abs/10.1021/jp9722824

    """

    def get_removed_dof(self, atoms):
        return 3

    def adjust_positions(self, atoms, new):
        masses = atoms.get_masses()
        old_cm = atoms.get_center_of_mass()
        new_cm = np.dot(masses, new) / masses.sum()
        d = old_cm - new_cm
        new += d

    def adjust_forces(self, atoms, forces):
        m = atoms.get_masses()
        mm = np.tile(m, (3, 1)).T
        lb = np.sum(mm * forces, axis=0) / sum(m ** 2)
        forces -= mm * lb

    def todict(self):
        return {'name': 'FixCom', 'kwargs': {}}