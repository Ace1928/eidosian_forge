from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
class TensorSymmetry(Basic):
    """
    Monoterm symmetry of a tensor (i.e. any symmetric or anti-symmetric
    index permutation). For the relevant terminology see ``tensor_can.py``
    section of the combinatorics module.

    Parameters
    ==========

    bsgs : tuple ``(base, sgs)`` BSGS of the symmetry of the tensor

    Attributes
    ==========

    ``base`` : base of the BSGS
    ``generators`` : generators of the BSGS
    ``rank`` : rank of the tensor

    Notes
    =====

    A tensor can have an arbitrary monoterm symmetry provided by its BSGS.
    Multiterm symmetries, like the cyclic symmetry of the Riemann tensor
    (i.e., Bianchi identity), are not covered. See combinatorics module for
    information on how to generate BSGS for a general index permutation group.
    Simple symmetries can be generated using built-in methods.

    See Also
    ========

    sympy.combinatorics.tensor_can.get_symmetric_group_sgs

    Examples
    ========

    Define a symmetric tensor of rank 2

    >>> from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> sym = TensorSymmetry(get_symmetric_group_sgs(2))
    >>> T = TensorHead('T', [Lorentz]*2, sym)

    Note, that the same can also be done using built-in TensorSymmetry methods

    >>> sym2 = TensorSymmetry.fully_symmetric(2)
    >>> sym == sym2
    True
    """

    def __new__(cls, *args, **kw_args):
        if len(args) == 1:
            base, generators = args[0]
        elif len(args) == 2:
            base, generators = args
        else:
            raise TypeError('bsgs required, either two separate parameters or one tuple')
        if not isinstance(base, Tuple):
            base = Tuple(*base)
        if not isinstance(generators, Tuple):
            generators = Tuple(*generators)
        return Basic.__new__(cls, base, generators, **kw_args)

    @property
    def base(self):
        return self.args[0]

    @property
    def generators(self):
        return self.args[1]

    @property
    def rank(self):
        return self.generators[0].size - 2

    @classmethod
    def fully_symmetric(cls, rank):
        """
        Returns a fully symmetric (antisymmetric if ``rank``<0)
        TensorSymmetry object for ``abs(rank)`` indices.
        """
        if rank > 0:
            bsgs = get_symmetric_group_sgs(rank, False)
        elif rank < 0:
            bsgs = get_symmetric_group_sgs(-rank, True)
        elif rank == 0:
            bsgs = ([], [Permutation(1)])
        return TensorSymmetry(bsgs)

    @classmethod
    def direct_product(cls, *args):
        """
        Returns a TensorSymmetry object that is being a direct product of
        fully (anti-)symmetric index permutation groups.

        Notes
        =====

        Some examples for different values of ``(*args)``:
        ``(1)``         vector, equivalent to ``TensorSymmetry.fully_symmetric(1)``
        ``(2)``         tensor with 2 symmetric indices, equivalent to ``.fully_symmetric(2)``
        ``(-2)``        tensor with 2 antisymmetric indices, equivalent to ``.fully_symmetric(-2)``
        ``(2, -2)``     tensor with the first 2 indices commuting and the last 2 anticommuting
        ``(1, 1, 1)``   tensor with 3 indices without any symmetry
        """
        base, sgs = ([], [Permutation(1)])
        for arg in args:
            if arg > 0:
                bsgs2 = get_symmetric_group_sgs(arg, False)
            elif arg < 0:
                bsgs2 = get_symmetric_group_sgs(-arg, True)
            else:
                continue
            base, sgs = bsgs_direct_product(base, sgs, *bsgs2)
        return TensorSymmetry(base, sgs)

    @classmethod
    def riemann(cls):
        """
        Returns a monotorem symmetry of the Riemann tensor
        """
        return TensorSymmetry(riemann_bsgs)

    @classmethod
    def no_symmetry(cls, rank):
        """
        TensorSymmetry object for ``rank`` indices with no symmetry
        """
        return TensorSymmetry([], [Permutation(rank + 1)])