from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
class Coset(Basic):
    """A left coset of a permutation group with respect to an element.

    Parameters
    ==========

    g : Permutation

    H : PermutationGroup

    dir : "+" or "-", If not specified by default it will be "+"
        here ``dir`` specified the type of coset "+" represent the
        right coset and "-" represent the left coset.

    G : PermutationGroup, optional
        The group which contains *H* as its subgroup and *g* as its
        element.

        If not specified, it would automatically become a symmetric
        group ``SymmetricPermutationGroup(g.size)`` and
        ``SymmetricPermutationGroup(H.degree)`` if ``g.size`` and ``H.degree``
        are matching.``SymmetricPermutationGroup`` is a lazy form of SymmetricGroup
        used for representation purpose.

    """

    def __new__(cls, g, H, G=None, dir='+'):
        g = _sympify(g)
        if not isinstance(g, Permutation):
            raise NotImplementedError
        H = _sympify(H)
        if not isinstance(H, PermutationGroup):
            raise NotImplementedError
        if G is not None:
            G = _sympify(G)
            if not isinstance(G, (PermutationGroup, SymmetricPermutationGroup)):
                raise NotImplementedError
            if not H.is_subgroup(G):
                raise ValueError('{} must be a subgroup of {}.'.format(H, G))
            if g not in G:
                raise ValueError('{} must be an element of {}.'.format(g, G))
        else:
            g_size = g.size
            h_degree = H.degree
            if g_size != h_degree:
                raise ValueError('The size of the permutation {} and the degree of the permutation group {} should be matching '.format(g, H))
            G = SymmetricPermutationGroup(g.size)
        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError('dir must be of type basestring or Symbol, not %s' % type(dir))
        if str(dir) not in ('+', '-'):
            raise ValueError("dir must be one of '+' or '-' not %s" % dir)
        obj = Basic.__new__(cls, g, H, G, dir)
        return obj

    def __init__(self, *args, **kwargs):
        self._dir = self.args[3]

    @property
    def is_left_coset(self):
        """
        Check if the coset is left coset that is ``gH``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="-")
        >>> cst.is_left_coset
        True

        """
        return str(self._dir) == '-'

    @property
    def is_right_coset(self):
        """
        Check if the coset is right coset that is ``Hg``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="+")
        >>> cst.is_right_coset
        True

        """
        return str(self._dir) == '+'

    def as_list(self):
        """
        Return all the elements of coset in the form of list.
        """
        g = self.args[0]
        H = self.args[1]
        cst = []
        if str(self._dir) == '+':
            for h in H.elements:
                cst.append(h * g)
        else:
            for h in H.elements:
                cst.append(g * h)
        return cst