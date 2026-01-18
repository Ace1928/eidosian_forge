from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class SquareTensor(Tensor):
    """Base class for doing useful general operations on second rank tensors
    (stress, strain etc.).
    """

    def __new__(cls, input_array, vscale=None) -> Self:
        """Create a SquareTensor object. Note that the constructor uses __new__ rather than
        __init__ according to the standard method of subclassing numpy ndarrays. Error
        is thrown when the class is initialized with non-square matrix.

        Args:
            input_array (3x3 array-like): the 3x3 array-like
                representing the content of the tensor
            vscale (6x1 array-like): 6x1 array-like scaling the
                Voigt-notation vector with the tensor entries
        """
        obj = super().__new__(cls, input_array, vscale, check_rank=2)
        return obj.view(cls)

    @property
    def trans(self):
        """Shorthand for transpose on SquareTensor."""
        return SquareTensor(np.transpose(self))

    @property
    def inv(self):
        """Shorthand for matrix inverse on SquareTensor."""
        if self.det == 0:
            raise ValueError('SquareTensor is non-invertible')
        return SquareTensor(np.linalg.inv(self))

    @property
    def det(self):
        """Shorthand for the determinant of the SquareTensor."""
        return np.linalg.det(self)

    def is_rotation(self, tol: float=0.001, include_improper=True):
        """Test to see if tensor is a valid rotation matrix, performs a
        test to check whether the inverse is equal to the transpose
        and if the determinant is equal to one within the specified
        tolerance.

        Args:
            tol (float): tolerance to both tests of whether the
                the determinant is one and the inverse is equal
                to the transpose
            include_improper (bool): whether to include improper
                rotations in the determination of validity
        """
        det = np.abs(np.linalg.det(self))
        if include_improper:
            det = np.abs(det)
        return (np.abs(self.inv - self.trans) < tol).all() and np.abs(det - 1.0) < tol

    def refine_rotation(self):
        """Helper method for refining rotation matrix by ensuring
        that second and third rows are perpendicular to the first.
        Gets new y vector from an orthogonal projection of x onto y
        and the new z vector from a cross product of the new x and y.

        Args:
            tol to test for rotation

        Returns:
            new rotation matrix
        """
        new_x, y = (get_uvec(self[0]), get_uvec(self[1]))
        new_y = y - np.dot(new_x, y) * new_x
        new_z = np.cross(new_x, new_y)
        return SquareTensor([new_x, new_y, new_z])

    def get_scaled(self, scale_factor):
        """Scales the tensor by a certain multiplicative scale factor.

        Args:
            scale_factor (float): scalar multiplier to be applied to the
                SquareTensor object
        """
        return SquareTensor(self * scale_factor)

    @property
    def principal_invariants(self):
        """Returns a list of principal invariants for the tensor,
        which are the values of the coefficients of the characteristic
        polynomial for the matrix.
        """
        return np.poly(self)[1:] * np.array([-1, 1, -1])

    def polar_decomposition(self, side='right'):
        """Calculates matrices for polar decomposition."""
        return polar(self, side=side)