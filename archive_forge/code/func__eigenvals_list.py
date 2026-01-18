from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def _eigenvals_list(M, error_when_incomplete=True, simplify=False, **flags):
    iblocks = M.strongly_connected_components()
    all_eigs = []
    is_dom = M._rep.domain in (ZZ, QQ)
    for b in iblocks:
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs.append(val)
            continue
        block = M[b, b]
        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()
        eigs = roots(charpoly, multiple=True, **flags)
        if len(eigs) != block.rows:
            degree = int(charpoly.degree())
            f = charpoly.as_expr()
            x = charpoly.gen
            try:
                eigs = [CRootOf(f, x, idx) for idx in range(degree)]
            except NotImplementedError:
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = []
        all_eigs += eigs
    if not simplify:
        return all_eigs
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    return [simplify(value) for value in all_eigs]