from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
class Strain(SquareTensor):
    """Subclass of SquareTensor that describes the Green-Lagrange strain tensor."""
    symbol = 'e'

    def __new__(cls, strain_matrix) -> Self:
        """
        Create a Strain object. Note that the constructor uses __new__
        rather than __init__ according to the standard method of
        subclassing numpy ndarrays. Note also that the default constructor
        does not include the deformation gradient.

        Args:
            strain_matrix (ArrayLike): 3x3 matrix or length-6 Voigt notation vector
                representing the Green-Lagrange strain
        """
        vscale = np.ones((6,))
        vscale[3:] *= 2
        obj = super().__new__(cls, strain_matrix, vscale=vscale)
        if not obj.is_symmetric():
            raise ValueError('Strain must be initialized with a symmetric array or a Voigt-notation vector with six entries.')
        return obj.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.rank = getattr(obj, 'rank', None)
        self._vscale = getattr(obj, '_vscale', None)

    @classmethod
    def from_deformation(cls, deformation: ArrayLike) -> Self:
        """
        Factory method that returns a Strain object from a deformation
        gradient.

        Args:
            deformation (ArrayLike): 3x3 array defining the deformation
        """
        dfm = Deformation(deformation)
        return cls(0.5 * (np.dot(dfm.trans, dfm) - np.eye(3)))

    @classmethod
    def from_index_amount(cls, idx: tuple | int, amount: float) -> Self:
        """
        Like Deformation.from_index_amount, except generates
        a strain from the zero 3x3 tensor or Voigt vector with
        the amount specified in the index location. Ensures
        symmetric strain.

        Args:
            idx (tuple or integer): index to be perturbed, can be Voigt or full-tensor notation
            amount (float): amount to perturb selected index
        """
        if isinstance(idx, int):
            v = np.zeros(6)
            v[idx] = amount
            return cls.from_voigt(v)
        if np.array(idx).ndim == 1:
            v = np.zeros((3, 3))
            for i in itertools.permutations(idx):
                v[i] = amount
            return cls(v)
        raise ValueError('Index must either be 2-tuple or integer corresponding to full-tensor or Voigt index')

    def get_deformation_matrix(self, shape: Literal['upper', 'lower', 'symmetric']='upper'):
        """
        Returns the deformation matrix.

        Args:
            shape ('upper' | 'lower' | 'symmetric'): method for determining deformation
                'upper' produces an upper triangular defo
                'lower' produces a lower triangular defo
                'symmetric' produces a symmetric defo
        """
        return convert_strain_to_deformation(self, shape=shape)

    @property
    def von_mises_strain(self):
        """Equivalent strain to Von Mises Stress."""
        eps = self - 1 / 3 * np.trace(self) * np.identity(3)
        return np.sqrt(np.sum(eps * eps) * 2 / 3)