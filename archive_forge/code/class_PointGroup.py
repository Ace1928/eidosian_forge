from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
@cached_class
class PointGroup(SymmetryGroup):
    """Class representing a Point Group, with generators and symmetry operations.

    Attributes:
        symbol (str): Full International or Hermann-Mauguin Symbol.
        generators (list): List of generator matrices. Note that 3x3 matrices are used for Point Groups.
        symmetry_ops (list): Full set of symmetry operations as matrices.
    """

    def __init__(self, int_symbol: str) -> None:
        """Initializes a Point Group from its international symbol.

        Args:
            int_symbol (str): International or Hermann-Mauguin Symbol.
        """
        from pymatgen.core.operations import SymmOp
        self.symbol = int_symbol
        self.generators = [SYMM_DATA['generator_matrices'][enc] for enc in SYMM_DATA['point_group_encoding'][int_symbol]]
        self._symmetry_ops = {SymmOp.from_rotation_and_translation(m) for m in self._generate_full_symmetry_ops()}
        self.order = len(self._symmetry_ops)

    @property
    def symmetry_ops(self) -> set[SymmOp]:
        """
        Returns:
            List of symmetry operations associated with the group.
        """
        return self._symmetry_ops

    def _generate_full_symmetry_ops(self) -> list[SymmOp]:
        symm_ops = list(self.generators)
        new_ops = self.generators
        while len(new_ops) > 0:
            gen_ops = []
            for g1, g2 in product(new_ops, symm_ops):
                op = np.dot(g1, g2)
                if not in_array_list(symm_ops, op):
                    gen_ops.append(op)
                    symm_ops.append(op)
            new_ops = gen_ops
        return symm_ops

    def get_orbit(self, p: ArrayLike, tol: float=1e-05) -> list[np.ndarray]:
        """Returns the orbit for a point.

        Args:
            p: Point as a 3x1 array.
            tol: Tolerance for determining if sites are the same. 1e-5 should
                be sufficient for most purposes. Set to 0 for exact matching
                (and also needed for symbolic orbits).

        Returns:
            list[array]: Orbit for point.
        """
        orbit: list[np.ndarray] = []
        for o in self.symmetry_ops:
            pp = o.operate(p)
            if not in_array_list(orbit, pp, tol=tol):
                orbit.append(pp)
        return orbit