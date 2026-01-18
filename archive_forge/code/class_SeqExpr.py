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
class SeqExpr(SeqBase):
    """Sequence expression class.

    Various sequences should inherit from this class.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExpr
    >>> from sympy.abc import x
    >>> from sympy import Tuple
    >>> s = SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, 10))
    >>> s.gen
    (1, 2, 3)
    >>> s.interval
    Interval(0, 10)
    >>> s.length
    11

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """

    @property
    def gen(self):
        return self.args[0]

    @property
    def interval(self):
        return Interval(self.args[1][1], self.args[1][2])

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return self.stop - self.start + 1

    @property
    def variables(self):
        return (self.args[1][0],)