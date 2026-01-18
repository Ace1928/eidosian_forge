from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
class MagSymmOp(SymmOp):
    """Thin wrapper around SymmOp to extend it to support magnetic symmetry by including a time
    reversal operator. Magnetic symmetry is similar to conventional crystal symmetry, except
    symmetry is reduced by the addition of a time reversal operator which acts on an atom's magnetic
    moment.
    """

    def __init__(self, affine_transformation_matrix: ArrayLike, time_reversal: int, tol: float=0.01) -> None:
        """Initializes the MagSymmOp from a 4x4 affine transformation matrix and time reversal
        operator. In general, this constructor should not be used unless you are transferring
        rotations. Use the static constructors instead to generate a SymmOp from proper rotations
        and translation.

        Args:
            affine_transformation_matrix (4x4 array): Representing an
                affine transformation.
            time_reversal (int): 1 or -1
            tol (float): Tolerance for determining if matrices are equal.
        """
        SymmOp.__init__(self, affine_transformation_matrix, tol=tol)
        if time_reversal in {-1, 1}:
            self.time_reversal = time_reversal
        else:
            raise RuntimeError(f'Invalid time_reversal={time_reversal!r}, must be 1 or -1')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymmOp):
            return NotImplemented
        return np.allclose(self.affine_matrix, other.affine_matrix, atol=self.tol) and self.time_reversal == other.time_reversal

    def __str__(self) -> str:
        return self.as_xyzt_str()

    def __repr__(self) -> str:
        output = ['Rot:', str(self.affine_matrix[0:3][:, 0:3]), 'tau', str(self.affine_matrix[0:3][:, 3]), 'Time reversal:', str(self.time_reversal)]
        return '\n'.join(output)

    def __hash__(self) -> int:
        hashable_value = (*tuple(self.affine_matrix.flatten()), self.time_reversal)
        return hash(hashable_value)

    @due.dcite(Doi('10.1051/epjconf/20122200010'), description='Symmetry and magnetic structures')
    def operate_magmom(self, magmom):
        """Apply time reversal operator on the magnetic moment. Note that
        magnetic moments transform as axial vectors, not polar vectors.

        See 'Symmetry and magnetic structures', Rodríguez-Carvajal and
        Bourée for a good discussion. DOI: 10.1051/epjconf/20122200010

        Args:
            magmom: Magnetic moment as electronic_structure.core.Magmom
            class or as list or np array-like

        Returns:
            Magnetic moment after operator applied as Magmom class
        """
        magmom = Magmom(magmom)
        transformed_moment = self.apply_rotation_only(magmom.global_moment) * np.linalg.det(self.rotation_matrix) * self.time_reversal
        return Magmom.from_global_moment_and_saxis(transformed_moment, magmom.saxis)

    @classmethod
    def from_symmop(cls, symmop: SymmOp, time_reversal) -> Self:
        """Initialize a MagSymmOp from a SymmOp and time reversal operator.

        Args:
            symmop (SymmOp): SymmOp
            time_reversal (int): Time reversal operator, +1 or -1.

        Returns:
            MagSymmOp object
        """
        return cls(symmop.affine_matrix, time_reversal, symmop.tol)

    @staticmethod
    def from_rotation_and_translation_and_time_reversal(rotation_matrix: ArrayLike=((1, 0, 0), (0, 1, 0), (0, 0, 1)), translation_vec: ArrayLike=(0, 0, 0), time_reversal: int=1, tol: float=0.1) -> MagSymmOp:
        """Creates a symmetry operation from a rotation matrix, translation
        vector and time reversal operator.

        Args:
            rotation_matrix (3x3 array): Rotation matrix.
            translation_vec (3x1 array): Translation vector.
            time_reversal (int): Time reversal operator, +1 or -1.
            tol (float): Tolerance to determine if rotation matrix is valid.

        Returns:
            MagSymmOp object
        """
        symm_op = SymmOp.from_rotation_and_translation(rotation_matrix=rotation_matrix, translation_vec=translation_vec, tol=tol)
        return MagSymmOp.from_symmop(symm_op, time_reversal)

    @classmethod
    def from_xyzt_str(cls, xyzt_str: str) -> Self:
        """
        Args:
            xyzt_str (str): of the form 'x, y, z, +1', '-x, -y, z, -1',
                '-2y+1/2, 3x+1/2, z-y+1/2, +1', etc.

        Returns:
            MagSymmOp object
        """
        symm_op = SymmOp.from_xyz_str(xyzt_str.rsplit(',', 1)[0])
        try:
            time_reversal = int(xyzt_str.rsplit(',', 1)[1])
        except Exception:
            raise RuntimeError('Time reversal operator could not be parsed.')
        return cls.from_symmop(symm_op, time_reversal)

    def as_xyzt_str(self) -> str:
        """Returns a string of the form 'x, y, z, +1', '-x, -y, z, -1',
        '-y+1/2, x+1/2, z+1/2, +1', etc. Only works for integer rotation matrices.
        """
        xyzt_string = SymmOp.as_xyz_str(self)
        return f'{xyzt_string}, {self.time_reversal:+}'

    def as_dict(self) -> dict[str, Any]:
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'matrix': self.affine_matrix.tolist(), 'tolerance': self.tol, 'time_reversal': self.time_reversal}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct: dict.

        Returns:
            MagneticSymmOp from dict representation.
        """
        return cls(dct['matrix'], tol=dct['tolerance'], time_reversal=dct['time_reversal'])