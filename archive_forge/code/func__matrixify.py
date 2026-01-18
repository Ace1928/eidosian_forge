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
def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""
    if getattr(mat, 'is_Matrix', False) or getattr(mat, 'is_MatrixLike', False):
        return mat
    if not (getattr(mat, 'is_Matrix', True) or getattr(mat, 'is_MatrixLike', True)):
        return mat
    shape = None
    if hasattr(mat, 'shape'):
        if len(mat.shape) == 2:
            shape = mat.shape
    elif hasattr(mat, 'rows') and hasattr(mat, 'cols'):
        shape = (mat.rows, mat.cols)
    if shape:
        return _MatrixWrapper(mat, shape)
    return mat