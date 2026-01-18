from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int
class FreeGroup(DefaultPrinting):
    """
    Free group with finite or infinite number of generators. Its input API
    is that of a str, Symbol/Expr or a sequence of one of
    these types (which may be empty)

    See Also
    ========

    sympy.polys.rings.PolyRing

    References
    ==========

    .. [1] https://www.gap-system.org/Manuals/doc/ref/chap37.html

    .. [2] https://en.wikipedia.org/wiki/Free_group

    """
    is_associative = True
    is_group = True
    is_FreeGroup = True
    is_PermutationGroup = False
    relators: list[Expr] = []

    def __new__(cls, symbols):
        symbols = tuple(_parse_symbols(symbols))
        rank = len(symbols)
        _hash = hash((cls.__name__, symbols, rank))
        obj = _free_group_cache.get(_hash)
        if obj is None:
            obj = object.__new__(cls)
            obj._hash = _hash
            obj._rank = rank
            obj.dtype = type('FreeGroupElement', (FreeGroupElement,), {'group': obj})
            obj.symbols = symbols
            obj.generators = obj._generators()
            obj._gens_set = set(obj.generators)
            for symbol, generator in zip(obj.symbols, obj.generators):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if hasattr(obj, name):
                        setattr(obj, name, generator)
            _free_group_cache[_hash] = obj
        return obj

    def _generators(group):
        """Returns the generators of the FreeGroup.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> F.generators
        (x, y, z)

        """
        gens = []
        for sym in group.symbols:
            elm = ((sym, 1),)
            gens.append(group.dtype(elm))
        return tuple(gens)

    def clone(self, symbols=None):
        return self.__class__(symbols or self.symbols)

    def __contains__(self, i):
        """Return True if ``i`` is contained in FreeGroup."""
        if not isinstance(i, FreeGroupElement):
            return False
        group = i.group
        return self == group

    def __hash__(self):
        return self._hash

    def __len__(self):
        return self.rank

    def __str__(self):
        if self.rank > 30:
            str_form = '<free group with %s generators>' % self.rank
        else:
            str_form = '<free group on the generators '
            gens = self.generators
            str_form += str(gens) + '>'
        return str_form
    __repr__ = __str__

    def __getitem__(self, index):
        symbols = self.symbols[index]
        return self.clone(symbols=symbols)

    def __eq__(self, other):
        """No ``FreeGroup`` is equal to any "other" ``FreeGroup``.
        """
        return self is other

    def index(self, gen):
        """Return the index of the generator `gen` from ``(f_0, ..., f_(n-1))``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.index(y)
        1
        >>> F.index(x)
        0

        """
        if isinstance(gen, self.dtype):
            return self.generators.index(gen)
        else:
            raise ValueError('expected a generator of Free Group %s, got %s' % (self, gen))

    def order(self):
        """Return the order of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.order()
        oo

        >>> free_group("")[0].order()
        1

        """
        if self.rank == 0:
            return S.One
        else:
            return S.Infinity

    @property
    def elements(self):
        """
        Return the elements of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> (z,) = free_group("")
        >>> z.elements
        {<identity>}

        """
        if self.rank == 0:
            return {self.identity}
        else:
            raise ValueError('Group contains infinitely many elements, hence cannot be represented')

    @property
    def rank(self):
        """
        In group theory, the `rank` of a group `G`, denoted `G.rank`,
        can refer to the smallest cardinality of a generating set
        for G, that is

        \\operatorname{rank}(G)=\\min\\{ |X|: X\\subseteq G, \\left\\langle X\\right\\rangle =G\\}.

        """
        return self._rank

    @property
    def is_abelian(self):
        """Returns if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.is_abelian
        False

        """
        return self.rank in (0, 1)

    @property
    def identity(self):
        """Returns the identity element of free group."""
        return self.dtype()

    def contains(self, g):
        """Tests if Free Group element ``g`` belong to self, ``G``.

        In mathematical terms any linear combination of generators
        of a Free Group is contained in it.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.contains(x**3*y**2)
        True

        """
        if not isinstance(g, FreeGroupElement):
            return False
        elif self != g.group:
            return False
        else:
            return True

    def center(self):
        """Returns the center of the free group `self`."""
        return {self.identity}