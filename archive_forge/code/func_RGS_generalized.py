from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int
from collections import defaultdict
def RGS_generalized(m):
    """
    Computes the m + 1 generalized unrestricted growth strings
    and returns them as rows in matrix.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_generalized
    >>> RGS_generalized(6)
    Matrix([
    [  1,   1,   1,  1,  1, 1, 1],
    [  1,   2,   3,  4,  5, 6, 0],
    [  2,   5,  10, 17, 26, 0, 0],
    [  5,  15,  37, 77,  0, 0, 0],
    [ 15,  52, 151,  0,  0, 0, 0],
    [ 52, 203,   0,  0,  0, 0, 0],
    [203,   0,   0,  0,  0, 0, 0]])
    """
    d = zeros(m + 1)
    for i in range(m + 1):
        d[0, i] = 1
    for i in range(1, m + 1):
        for j in range(m):
            if j <= m - i:
                d[i, j] = j * d[i - 1, j] + d[i - 1, j + 1]
            else:
                d[i, j] = 0
    return d