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
class SeqFormula(SeqExpr):
    """
    Represents sequence based on a formula.

    Elements are generated using a formula.

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    For value at a particular point

    >>> s.coeff(3)
    9

    supports slicing

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    iterable

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    sequence starts from negative infinity

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """

    def __new__(cls, formula, limits=None):
        formula = sympify(formula)

        def _find_x(formula):
            free = formula.free_symbols
            if len(free) == 1:
                return free.pop()
            elif not free:
                return Dummy('k')
            else:
                raise ValueError(' specify dummy variables for %s. If the formula contains more than one free symbol, a dummy variable should be supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))' % formula)
        x, start, stop = (None, None, None)
        if limits is None:
            x, start, stop = (_find_x(formula), 0, S.Infinity)
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits
        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))
        if start is S.NegativeInfinity and stop is S.Infinity:
            raise ValueError('Both the start and end value cannot be unbounded')
        limits = sympify((x, start, stop))
        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence
        return Basic.__new__(cls, formula, limits)

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqFormula):
            form1, v1 = (self.formula, self.variables[0])
            form2, v2 = (other.formula, other.variables[0])
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqFormula):
            form1, v1 = (self.formula, self.variables[0])
            form2, v2 = (other.formula, other.variables[0])
            formula = form1 * form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        formula = self.formula * coeff
        return SeqFormula(formula, self.args[1])

    def expand(self, *args, **kwargs):
        return SeqFormula(expand(self.formula, *args, **kwargs), self.args[1])