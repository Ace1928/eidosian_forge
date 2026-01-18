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
class _MatrixWrapper:
    """Wrapper class providing the minimum functionality for a matrix-like
    object: .rows, .cols, .shape, indexability, and iterability. CommonMatrix
    math operations should work on matrix-like objects. This one is intended for
    matrix-like objects which use the same indexing format as SymPy with respect
    to returning matrix elements instead of rows for non-tuple indexes.
    """
    is_Matrix = False
    is_MatrixLike = True

    def __init__(self, mat, shape):
        self.mat = mat
        self.shape = shape
        self.rows, self.cols = shape

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return sympify(self.mat.__getitem__(key))
        return sympify(self.mat.__getitem__((key // self.rows, key % self.cols)))

    def __iter__(self):
        mat = self.mat
        cols = self.cols
        return iter((sympify(mat[r, c]) for r in range(self.rows) for c in range(cols)))