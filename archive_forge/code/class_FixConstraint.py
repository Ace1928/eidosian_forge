from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixConstraint:
    """Base class for classes that fix one or more atoms in some way."""

    def index_shuffle(self, atoms, ind):
        """Change the indices.

        When the ordering of the atoms in the Atoms object changes,
        this method can be called to shuffle the indices of the
        constraints.

        ind -- List or tuple of indices.

        """
        raise NotImplementedError

    def repeat(self, m, n):
        """ basic method to multiply by m, needs to know the length
        of the underlying atoms object for the assignment of
        multiplied constraints to work.
        """
        msg = "Repeat is not compatible with your atoms' constraints. Use atoms.set_constraint() before calling repeat to remove your constraints."
        raise NotImplementedError(msg)

    def adjust_momenta(self, atoms, momenta):
        """Adjusts momenta in identical manner to forces."""
        self.adjust_forces(atoms, momenta)

    def copy(self):
        return dict2constraint(self.todict().copy())