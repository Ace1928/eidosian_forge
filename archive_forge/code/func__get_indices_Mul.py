from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def _get_indices_Mul(expr, return_dummies=False):
    """Determine the outer indices of a Mul object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Mul
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Mul(x[i, k]*y[j, k])
    ({i, j}, {})
    >>> _get_indices_Mul(x[i, k]*y[j, k], return_dummies=True)
    ({i, j}, {}, (k,))

    """
    inds = list(map(get_indices, expr.args))
    inds, syms = list(zip(*inds))
    inds = list(map(list, inds))
    inds = list(reduce(lambda x, y: x + y, inds))
    inds, dummies = _remove_repeated(inds)
    symmetry = {}
    for s in syms:
        for pair in s:
            if pair in symmetry:
                symmetry[pair] *= s[pair]
            else:
                symmetry[pair] = s[pair]
    if return_dummies:
        return (inds, symmetry, dummies)
    else:
        return (inds, symmetry)