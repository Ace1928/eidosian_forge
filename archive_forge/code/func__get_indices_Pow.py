from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def _get_indices_Pow(expr):
    """Determine outer indices of a power or an exponential.

    A power is considered a universal function, so that the indices of a Pow is
    just the collection of indices present in the expression.  This may be
    viewed as a bit inconsistent in the special case:

        x[i]**2 = x[i]*x[i]                                                      (1)

    The above expression could have been interpreted as the contraction of x[i]
    with itself, but we choose instead to interpret it as a function

        lambda y: y**2

    applied to each element of x (a universal function in numpy terms).  In
    order to allow an interpretation of (1) as a contraction, we need
    contravariant and covariant Idx subclasses.  (FIXME: this is not yet
    implemented)

    Expressions in the base or exponent are subject to contraction as usual,
    but an index that is present in the exponent, will not be considered
    contractable with its own base.  Note however, that indices in the same
    exponent can be contracted with each other.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Pow
    >>> from sympy import Pow, exp, IndexedBase, Idx
    >>> A = IndexedBase('A')
    >>> x = IndexedBase('x')
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> _get_indices_Pow(exp(A[i, j]*x[j]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(x[i], x[i]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(A[i, j]*x[j], x[i]))
    ({i}, {})

    """
    base, exp = expr.as_base_exp()
    binds, bsyms = get_indices(base)
    einds, esyms = get_indices(exp)
    inds = binds | einds
    symmetries = {}
    return (inds, symmetries)