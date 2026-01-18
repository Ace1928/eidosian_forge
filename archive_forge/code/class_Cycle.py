import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
class Cycle(dict):
    """
    Wrapper around dict which provides the functionality of a disjoint cycle.

    Explanation
    ===========

    A cycle shows the rule to use to move subsets of elements to obtain
    a permutation. The Cycle class is more flexible than Permutation in
    that 1) all elements need not be present in order to investigate how
    multiple cycles act in sequence and 2) it can contain singletons:

    >>> from sympy.combinatorics.permutations import Perm, Cycle

    A Cycle will automatically parse a cycle given as a tuple on the rhs:

    >>> Cycle(1, 2)(2, 3)
    (1 3 2)

    The identity cycle, Cycle(), can be used to start a product:

    >>> Cycle()(1, 2)(2, 3)
    (1 3 2)

    The array form of a Cycle can be obtained by calling the list
    method (or passing it to the list function) and all elements from
    0 will be shown:

    >>> a = Cycle(1, 2)
    >>> a.list()
    [0, 2, 1]
    >>> list(a)
    [0, 2, 1]

    If a larger (or smaller) range is desired use the list method and
    provide the desired size -- but the Cycle cannot be truncated to
    a size smaller than the largest element that is out of place:

    >>> b = Cycle(2, 4)(1, 2)(3, 1, 4)(1, 3)
    >>> b.list()
    [0, 2, 1, 3, 4]
    >>> b.list(b.size + 1)
    [0, 2, 1, 3, 4, 5]
    >>> b.list(-1)
    [0, 2, 1]

    Singletons are not shown when printing with one exception: the largest
    element is always shown -- as a singleton if necessary:

    >>> Cycle(1, 4, 10)(4, 5)
    (1 5 4 10)
    >>> Cycle(1, 2)(4)(5)(10)
    (1 2)(10)

    The array form can be used to instantiate a Permutation so other
    properties of the permutation can be investigated:

    >>> Perm(Cycle(1, 2)(3, 4).list()).transpositions()
    [(1, 2), (3, 4)]

    Notes
    =====

    The underlying structure of the Cycle is a dictionary and although
    the __iter__ method has been redefined to give the array form of the
    cycle, the underlying dictionary items are still available with the
    such methods as items():

    >>> list(Cycle(1, 2).items())
    [(1, 2), (2, 1)]

    See Also
    ========

    Permutation
    """

    def __missing__(self, arg):
        """Enter arg into dictionary and return arg."""
        return as_int(arg)

    def __iter__(self):
        yield from self.list()

    def __call__(self, *other):
        """Return product of cycles processed from R to L.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        An instance of a Cycle will automatically parse list-like
        objects and Permutations that are on the right. It is more
        flexible than the Permutation in that all elements need not
        be present:

        >>> a = Cycle(1, 2)
        >>> a(2, 3)
        (1 3 2)
        >>> a(2, 3)(4, 5)
        (1 3 2)(4 5)

        """
        rv = Cycle(*other)
        for k, v in zip(list(self.keys()), [rv[self[k]] for k in self.keys()]):
            rv[k] = v
        return rv

    def list(self, size=None):
        """Return the cycles as an explicit list starting from 0 up
        to the greater of the largest value in the cycles and size.

        Truncation of trailing unmoved items will occur when size
        is less than the maximum element in the cycle; if this is
        desired, setting ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> p = Cycle(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Cycle(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        """
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        if size is not None:
            big = max([i for i in self.keys() if self[i] != i] + [0])
            size = max(size, big + 1)
        else:
            size = self.size
        return [self[i] for i in range(size)]

    def __repr__(self):
        """We want it to print as a Cycle, not as a dict.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> print(_)
        (1 2)
        >>> list(Cycle(1, 2).items())
        [(1, 2), (2, 1)]
        """
        if not self:
            return 'Cycle()'
        cycles = Permutation(self).cyclic_form
        s = ''.join((str(tuple(c)) for c in cycles))
        big = self.size - 1
        if not any((i == big for c in cycles for i in c)):
            s += '(%s)' % big
        return 'Cycle%s' % s

    def __str__(self):
        """We want it to be printed in a Cycle notation with no
        comma in-between.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> Cycle(1, 2, 4)(5, 6)
        (1 2 4)(5 6)
        """
        if not self:
            return '()'
        cycles = Permutation(self).cyclic_form
        s = ''.join((str(tuple(c)) for c in cycles))
        big = self.size - 1
        if not any((i == big for c in cycles for i in c)):
            s += '(%s)' % big
        s = s.replace(',', '')
        return s

    def __init__(self, *args):
        """Load up a Cycle instance with the values for the cycle.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2, 6)
        (1 2 6)
        """
        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], Permutation):
                for c in args[0].cyclic_form:
                    self.update(self(*c))
                return
            elif isinstance(args[0], Cycle):
                for k, v in args[0].items():
                    self[k] = v
                return
        args = [as_int(a) for a in args]
        if any((i < 0 for i in args)):
            raise ValueError('negative integers are not allowed in a cycle.')
        if has_dups(args):
            raise ValueError('All elements must be unique in a cycle.')
        for i in range(-len(args), 0):
            self[args[i]] = args[i + 1]

    @property
    def size(self):
        if not self:
            return 0
        return max(self.keys()) + 1

    def copy(self):
        return Cycle(self)