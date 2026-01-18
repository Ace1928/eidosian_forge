from itertools import combinations_with_replacement, product
from textwrap import dedent
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable
@public
class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """
    __slots__ = ('exponents', 'gens')

    def __init__(self, monom, gens=None):
        if not iterable(monom):
            rep, gens = dict_from_expr(sympify(monom), gens=gens)
            if len(rep) == 1 and list(rep.values())[0] == 1:
                monom = list(rep.keys())[0]
            else:
                raise ValueError('Expected a monomial got {}'.format(monom))
        self.exponents = tuple(map(int, monom))
        self.gens = gens

    def rebuild(self, exponents, gens=None):
        return self.__class__(exponents, gens or self.gens)

    def __len__(self):
        return len(self.exponents)

    def __iter__(self):
        return iter(self.exponents)

    def __getitem__(self, item):
        return self.exponents[item]

    def __hash__(self):
        return hash((self.__class__.__name__, self.exponents, self.gens))

    def __str__(self):
        if self.gens:
            return '*'.join(['%s**%s' % (gen, exp) for gen, exp in zip(self.gens, self.exponents)])
        else:
            return '%s(%s)' % (self.__class__.__name__, self.exponents)

    def as_expr(self, *gens):
        """Convert a monomial instance to a SymPy expression. """
        gens = gens or self.gens
        if not gens:
            raise ValueError('Cannot convert %s to an expression without generators' % self)
        return Mul(*[gen ** exp for gen, exp in zip(gens, self.exponents)])

    def __eq__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            return False
        return self.exponents == exponents

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError
        return self.rebuild(monomial_mul(self.exponents, exponents))

    def __truediv__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError
        result = monomial_div(self.exponents, exponents)
        if result is not None:
            return self.rebuild(result)
        else:
            raise ExactQuotientFailed(self, Monomial(other))
    __floordiv__ = __truediv__

    def __pow__(self, other):
        n = int(other)
        if not n:
            return self.rebuild([0] * len(self))
        elif n > 0:
            exponents = self.exponents
            for i in range(1, n):
                exponents = monomial_mul(exponents, self.exponents)
            return self.rebuild(exponents)
        else:
            raise ValueError('a non-negative integer expected, got %s' % other)

    def gcd(self, other):
        """Greatest common divisor of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError('an instance of Monomial class expected, got %s' % other)
        return self.rebuild(monomial_gcd(self.exponents, exponents))

    def lcm(self, other):
        """Least common multiple of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError('an instance of Monomial class expected, got %s' % other)
        return self.rebuild(monomial_lcm(self.exponents, exponents))