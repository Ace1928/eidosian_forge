from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce
from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.kind import Kind, NumberKind
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray
from .utilities import _get_intermediate_simp_bool
@classmethod
def _eval_jordan_block(cls, size: int, eigenvalue, band='upper'):
    if band == 'lower':

        def entry(i, j):
            if i == j:
                return eigenvalue
            elif j + 1 == i:
                return cls.one
            return cls.zero
    else:

        def entry(i, j):
            if i == j:
                return eigenvalue
            elif i + 1 == j:
                return cls.one
            return cls.zero
    return cls._new(size, size, entry)