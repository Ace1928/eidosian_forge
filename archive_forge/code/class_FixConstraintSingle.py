from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixConstraintSingle(FixConstraint):
    """Base class for classes that fix a single atom."""

    def __init__(self, a):
        self.a = a

    def index_shuffle(self, atoms, ind):
        """The atom index must be stored as self.a."""
        newa = None
        if self.a < 0:
            self.a += len(atoms)
        for new, old in slice2enlist(ind, len(atoms)):
            if old == self.a:
                newa = new
                break
        if newa is None:
            raise IndexError('Constraint not part of slice')
        self.a = newa

    def get_indices(self):
        return [self.a]