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
class RecursiveSeq(SeqBase):
    """
    A finite degree recursive sequence.

    Explanation
    ===========

    That is, a sequence a(n) that depends on a fixed, finite number of its
    previous values. The general form is

        a(n) = f(a(n - 1), a(n - 2), ..., a(n - d))

    for some fixed, positive integer d, where f is some function defined by a
    SymPy expression.

    Parameters
    ==========

    recurrence : SymPy expression defining recurrence
        This is *not* an equality, only the expression that the nth term is
        equal to. For example, if :code:`a(n) = f(a(n - 1), ..., a(n - d))`,
        then the expression should be :code:`f(a(n - 1), ..., a(n - d))`.

    yn : applied undefined function
        Represents the nth term of the sequence as e.g. :code:`y(n)` where
        :code:`y` is an undefined function and `n` is the sequence index.

    n : symbolic argument
        The name of the variable that the recurrence is in, e.g., :code:`n` if
        the recurrence function is :code:`y(n)`.

    initial : iterable with length equal to the degree of the recurrence
        The initial values of the recurrence.

    start : start value of sequence (inclusive)

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.series.sequences import RecursiveSeq
    >>> y = Function("y")
    >>> n = symbols("n")
    >>> fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])

    >>> fib.coeff(3) # Value at a particular point
    2

    >>> fib[:6] # supports slicing
    [0, 1, 1, 2, 3, 5]

    >>> fib.recurrence # inspect recurrence
    Eq(y(n), y(n - 2) + y(n - 1))

    >>> fib.degree # automatically determine degree
    2

    >>> for x in zip(range(10), fib): # supports iteration
    ...     print(x)
    (0, 0)
    (1, 1)
    (2, 1)
    (3, 2)
    (4, 3)
    (5, 5)
    (6, 8)
    (7, 13)
    (8, 21)
    (9, 34)

    See Also
    ========

    sympy.series.sequences.SeqFormula

    """

    def __new__(cls, recurrence, yn, n, initial=None, start=0):
        if not isinstance(yn, AppliedUndef):
            raise TypeError('recurrence sequence must be an applied undefined function, found `{}`'.format(yn))
        if not isinstance(n, Basic) or not n.is_symbol:
            raise TypeError('recurrence variable must be a symbol, found `{}`'.format(n))
        if yn.args != (n,):
            raise TypeError('recurrence sequence does not match symbol')
        y = yn.func
        k = Wild('k', exclude=(n,))
        degree = 0
        prev_ys = recurrence.find(y)
        for prev_y in prev_ys:
            if len(prev_y.args) != 1:
                raise TypeError('Recurrence should be in a single variable')
            shift = prev_y.args[0].match(n + k)[k]
            if not (shift.is_constant() and shift.is_integer and (shift < 0)):
                raise TypeError('Recurrence should have constant, negative, integer shifts (found {})'.format(prev_y))
            if -shift > degree:
                degree = -shift
        if not initial:
            initial = [Dummy('c_{}'.format(k)) for k in range(degree)]
        if len(initial) != degree:
            raise ValueError('Number of initial terms must equal degree')
        degree = Integer(degree)
        start = sympify(start)
        initial = Tuple(*(sympify(x) for x in initial))
        seq = Basic.__new__(cls, recurrence, yn, n, initial, start)
        seq.cache = {y(start + k): init for k, init in enumerate(initial)}
        seq.degree = degree
        return seq

    @property
    def _recurrence(self):
        """Equation defining recurrence."""
        return self.args[0]

    @property
    def recurrence(self):
        """Equation defining recurrence."""
        return Eq(self.yn, self.args[0])

    @property
    def yn(self):
        """Applied function representing the nth term"""
        return self.args[1]

    @property
    def y(self):
        """Undefined function for the nth term of the sequence"""
        return self.yn.func

    @property
    def n(self):
        """Sequence index symbol"""
        return self.args[2]

    @property
    def initial(self):
        """The initial values of the sequence"""
        return self.args[3]

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        return self.args[4]

    @property
    def stop(self):
        """The ending point of the sequence. (oo)"""
        return S.Infinity

    @property
    def interval(self):
        """Interval on which sequence is defined."""
        return (self.start, S.Infinity)

    def _eval_coeff(self, index):
        if index - self.start < len(self.cache):
            return self.cache[self.y(index)]
        for current in range(len(self.cache), index + 1):
            seq_index = self.start + current
            current_recurrence = self._recurrence.xreplace({self.n: seq_index})
            new_term = current_recurrence.xreplace(self.cache)
            self.cache[self.y(seq_index)] = new_term
        return self.cache[self.y(self.start + current)]

    def __iter__(self):
        index = self.start
        while True:
            yield self._eval_coeff(index)
            index += 1