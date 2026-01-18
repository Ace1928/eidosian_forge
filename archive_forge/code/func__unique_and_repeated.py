from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def _unique_and_repeated(inds):
    """
    Returns the unique and repeated indices. Also note, from the examples given below
    that the order of indices is maintained as given in the input.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _unique_and_repeated
    >>> _unique_and_repeated([2, 3, 1, 3, 0, 4, 0])
    ([2, 1, 4], [3, 0])
    """
    uniq = OrderedDict()
    for i in inds:
        if i in uniq:
            uniq[i] = 0
        else:
            uniq[i] = 1
    return sift(uniq, lambda x: uniq[x], binary=True)