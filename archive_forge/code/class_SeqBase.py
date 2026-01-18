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
class SeqBase(Basic):
    """Base class for sequences"""
    is_commutative = True
    _op_priority = 15

    @staticmethod
    def _start_key(expr):
        """Return start (if possible) else S.Infinity.

        adapted from Set._infimum_key
        """
        try:
            start = expr.start
        except (NotImplementedError, AttributeError, ValueError):
            start = S.Infinity
        return start

    def _intersect_interval(self, other):
        """Returns start and stop.

        Takes intersection over the two intervals.
        """
        interval = Intersection(self.interval, other.interval)
        return (interval.inf, interval.sup)

    @property
    def gen(self):
        """Returns the generator for the sequence"""
        raise NotImplementedError('(%s).gen' % self)

    @property
    def interval(self):
        """The interval on which the sequence is defined"""
        raise NotImplementedError('(%s).interval' % self)

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        raise NotImplementedError('(%s).start' % self)

    @property
    def stop(self):
        """The ending point of the sequence. This point is included"""
        raise NotImplementedError('(%s).stop' % self)

    @property
    def length(self):
        """Length of the sequence"""
        raise NotImplementedError('(%s).length' % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n, m
        >>> SeqFormula(m*n**2, (n, 0, 5)).free_symbols
        {m}
        """
        return {j for i in self.args for j in i.free_symbols.difference(self.variables)}

    @cacheit
    def coeff(self, pt):
        """Returns the coefficient at point pt"""
        if pt < self.start or pt > self.stop:
            raise IndexError('Index %s out of bounds %s' % (pt, self.interval))
        return self._eval_coeff(pt)

    def _eval_coeff(self, pt):
        raise NotImplementedError('The _eval_coeff method should be added to%s to return coefficient so it is availablewhen coeff calls it.' % self.func)

    def _ith_point(self, i):
        """Returns the i'th point of a sequence.

        Explanation
        ===========

        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        =========

        >>> from sympy import oo
        >>> from sympy.series.sequences import SeqPer

        bounded

        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(0)
        -10
        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(5)
        -5

        End is at infinity

        >>> SeqPer((1, 2, 3), (0, oo))._ith_point(5)
        5

        Starts at negative infinity

        >>> SeqPer((1, 2, 3), (-oo, 0))._ith_point(5)
        -5
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
        else:
            initial = self.start
        if self.start is S.NegativeInfinity:
            step = -1
        else:
            step = 1
        return initial + i * step

    def _add(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._add(other) returns a new, term-wise added sequence if self
        knows how to add with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqAdd` class.
        """
        return None

    def _mul(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._mul(other) returns a new, term-wise multiplied sequence if self
        knows how to multiply with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqMul` class.
        """
        return None

    def coeff_mul(self, other):
        """
        Should be used when ``other`` is not a sequence. Should be
        defined to define custom behaviour.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' defines multiplication of sequences with sequences only.
        """
        return Mul(self, other)

    def __add__(self, other):
        """Returns the term-wise addition of 'self' and 'other'.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) + SeqFormula(n**3)
        SeqFormula(n**3 + n**2, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return SeqAdd(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Returns the term-wise subtraction of ``self`` and ``other``.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) - (SeqFormula(n))
        SeqFormula(n**2 - n, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return SeqAdd(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        """Negates the sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)

    def __mul__(self, other):
        """Returns the term-wise multiplication of 'self' and 'other'.

        ``other`` should be a sequence. For ``other`` not being a
        sequence see :func:`coeff_mul` method.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) * (SeqFormula(n))
        SeqFormula(n**3, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return SeqMul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self * other

    def __iter__(self):
        for i in range(self.length):
            pt = self._ith_point(i)
            yield self.coeff(pt)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.coeff(index)
        elif isinstance(index, slice):
            start, stop = (index.start, index.stop)
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.coeff(self._ith_point(i)) for i in range(start, stop, index.step or 1)]

    def find_linear_recurrence(self, n, d=None, gfvar=None):
        """
        Finds the shortest linear recurrence that satisfies the first n
        terms of sequence of order `\\leq` ``n/2`` if possible.
        If ``d`` is specified, find shortest linear recurrence of order
        `\\leq` min(d, n/2) if possible.
        Returns list of coefficients ``[b(1), b(2), ...]`` corresponding to the
        recurrence relation ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        Returns ``[]`` if no recurrence is found.
        If gfvar is specified, also returns ordinary generating function as a
        function of gfvar.

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
        from sympy.simplify import simplify
        x = [simplify(expand(t)) for t in self[:n]]
        lx = len(x)
        if d is None:
            r = lx // 2
        else:
            r = min(d, lx // 2)
        coeffs = []
        for l in range(1, r + 1):
            l2 = 2 * l
            mlist = []
            for k in range(l):
                mlist.append(x[k:k + l])
            m = Matrix(mlist)
            if m.det() != 0:
                y = simplify(m.LUsolve(Matrix(x[l:l2])))
                if lx == l2:
                    coeffs = flatten(y[::-1])
                    break
                mlist = []
                for k in range(l, lx - l):
                    mlist.append(x[k:k + l])
                m = Matrix(mlist)
                if m * y == Matrix(x[l2:]):
                    coeffs = flatten(y[::-1])
                    break
        if gfvar is None:
            return coeffs
        else:
            l = len(coeffs)
            if l == 0:
                return ([], None)
            else:
                n, d = (x[l - 1] * gfvar ** (l - 1), 1 - coeffs[l - 1] * gfvar ** l)
                for i in range(l - 1):
                    n += x[i] * gfvar ** i
                    for j in range(l - i - 1):
                        n -= coeffs[i] * x[j] * gfvar ** (i + j + 1)
                    d -= coeffs[i] * gfvar ** (i + 1)
                return (coeffs, simplify(factor(n) / factor(d)))