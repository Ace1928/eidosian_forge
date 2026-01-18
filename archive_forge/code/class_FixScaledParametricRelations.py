from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixScaledParametricRelations(FixParametricRelations):

    def __init__(self, indices, Jacobian, const_shift, params=None, eps=1e-12):
        """The fractional coordinate version of FixParametricRelations

        All arguments are the same, but since this is for fractional coordinates use_cell is false
        """
        super(FixScaledParametricRelations, self).__init__(indices, Jacobian, const_shift, params, eps, False)

    def adjust_contravariant(self, cell, vecs, B):
        """Adjust the values of a set of vectors that are contravariant with the unit transformation"""
        scaled = cell.scaled_positions(vecs).flatten()
        scaled = self.Jacobian_inv @ (scaled - B)
        scaled = (self.Jacobian @ scaled + B).reshape((-1, 3))
        return cell.cartesian_positions(scaled)

    def adjust_positions(self, atoms, positions):
        """Adjust positions of the atoms to match the constraints"""
        positions[self.indices] = self.adjust_contravariant(atoms.cell, positions[self.indices], self.const_shift)
        positions[self.indices] = self.adjust_B(atoms.cell, positions[self.indices])

    def adjust_B(self, cell, positions):
        """Wraps the positions back to the unit cell and adjust B to keep track of this change"""
        fractional = cell.scaled_positions(positions)
        wrapped_fractional = fractional % 1.0 % 1.0
        self.const_shift += np.round(wrapped_fractional - fractional).flatten()
        return cell.cartesian_positions(wrapped_fractional)

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta of the atoms to match the constraints"""
        momenta[self.indices] = self.adjust_contravariant(atoms.cell, momenta[self.indices], np.zeros(self.const_shift.shape))

    def adjust_forces(self, atoms, forces):
        """Adjust forces of the atoms to match the constraints"""
        cart2frac_jacob = np.zeros(2 * (3 * len(atoms),))
        for i_atom in range(len(atoms)):
            cart2frac_jacob[3 * i_atom:3 * (i_atom + 1), 3 * i_atom:3 * (i_atom + 1)] = atoms.cell.T
        jacobian = cart2frac_jacob @ self.Jacobian
        jacobian_inv = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T
        reduced_forces = jacobian.T @ forces.flatten()
        forces[self.indices] = (jacobian_inv.T @ reduced_forces).reshape(-1, 3)

    def todict(self):
        """Create a dictionary representation of the constraint"""
        dct = super(FixScaledParametricRelations, self).todict()
        del dct['kwargs']['use_cell']
        return dct