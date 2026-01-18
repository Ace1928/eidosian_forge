from sympy.core import S, sympify
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int

    Returns the intersecting product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The intersecting product of given sequences is the sequence which
    contains the sum of products of the elements of the given sequences
    grouped by the *bitwise-AND* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which intersecting product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, intersecting_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> intersecting_product([u, v], [x, y])
    [u*x + u*y + v*x, v*y]
    >>> intersecting_product([u, v, x], [y, z])
    [u*y + u*z + v*y + x*y + x*z, v*z, 0, 0]

    >>> intersecting_product([1, S(2)/3], [3, 4 + 5*I])
    [9 + 5*I, 8/3 + 10*I/3]
    >>> intersecting_product([1, 3, S(5)/7], [7, 8])
    [327/7, 24, 0, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    