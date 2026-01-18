from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def _get_indices_Add(expr):
    """Determine outer indices of an Add object.

    In a sum, each term must have the same set of outer indices.  A valid
    expression could be

        x(i)*y(j) - x(j)*y(i)

    But we do not allow expressions like:

        x(i)*y(j) - z(j)*z(j)

    FIXME: Add support for Numpy broadcasting

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Add
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Add(x[i] + x[k]*y[i, k])
    ({i}, {})

    """
    inds = list(map(get_indices, expr.args))
    inds, syms = list(zip(*inds))
    non_scalars = [x for x in inds if x != set()]
    if not non_scalars:
        return (set(), {})
    if not all((x == non_scalars[0] for x in non_scalars[1:])):
        raise IndexConformanceException('Indices are not consistent: %s' % expr)
    if not reduce(lambda x, y: x != y or y, syms):
        symmetries = syms[0]
    else:
        symmetries = {}
    return (non_scalars[0], symmetries)