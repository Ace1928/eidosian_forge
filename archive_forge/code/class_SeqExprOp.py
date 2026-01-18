from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable
class SeqExprOp(SeqBase):
    """
    Base class for operations on sequences.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExprOp, sequence
    >>> from sympy.abc import n
    >>> s1 = sequence(n**2, (n, 0, 10))
    >>> s2 = sequence((1, 2, 3), (n, 5, 10))
    >>> s = SeqExprOp(s1, s2)
    >>> s.gen
    (n**2, (1, 2, 3))
    >>> s.interval
    Interval(5, 10)
    >>> s.length
    6

    See Also
    ========

    sympy.series.sequences.SeqAdd
    sympy.series.sequences.SeqMul
    """

    @property
    def gen(self):
        """Generator for the sequence.

        returns a tuple of generators of all the argument sequences.
        """
        return tuple((a.gen for a in self.args))

    @property
    def interval(self):
        """Sequence is defined on the intersection
        of all the intervals of respective sequences
        """
        return Intersection(*(a.interval for a in self.args))

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def variables(self):
        """Cumulative of all the bound variables"""
        return tuple(flatten([a.variables for a in self.args]))

    @property
    def length(self):
        return self.stop - self.start + 1