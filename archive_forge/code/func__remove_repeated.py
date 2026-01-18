from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def _remove_repeated(inds):
    """
    Removes repeated objects from sequences

    Returns a set of the unique objects and a tuple of all that have been
    removed.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _remove_repeated
    >>> l1 = [1, 2, 3, 2]
    >>> _remove_repeated(l1)
    ({1, 3}, (2,))

    """
    u, r = _unique_and_repeated(inds)
    return (set(u), tuple(r))