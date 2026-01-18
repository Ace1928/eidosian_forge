from collections.abc import Iterable
from sympy.core.numbers import Number
from sympy.core.assumptions import StdFactKB
from sympy.core import Expr, Tuple, sympify, S
from sympy.core.symbol import _filter_assumptions, Symbol
from sympy.core.logic import fuzzy_bool, fuzzy_not
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.multipledispatch import dispatch
from sympy.utilities.iterables import is_sequence, NotIterable
from sympy.utilities.misc import filldedent
class Idx(Expr):
    """Represents an integer index as an ``Integer`` or integer expression.

    There are a number of ways to create an ``Idx`` object.  The constructor
    takes two arguments:

    ``label``
        An integer or a symbol that labels the index.
    ``range``
        Optionally you can specify a range as either

        * ``Symbol`` or integer: This is interpreted as a dimension. Lower and
          upper bounds are set to ``0`` and ``range - 1``, respectively.
        * ``tuple``: The two elements are interpreted as the lower and upper
          bounds of the range, respectively.

    Note: bounds of the range are assumed to be either integer or infinite (oo
    and -oo are allowed to specify an unbounded range). If ``n`` is given as a
    bound, then ``n.is_integer`` must not return false.

    For convenience, if the label is given as a string it is automatically
    converted to an integer symbol.  (Note: this conversion is not done for
    range or dimension arguments.)

    Examples
    ========

    >>> from sympy import Idx, symbols, oo
    >>> n, i, L, U = symbols('n i L U', integer=True)

    If a string is given for the label an integer ``Symbol`` is created and the
    bounds are both ``None``:

    >>> idx = Idx('qwerty'); idx
    qwerty
    >>> idx.lower, idx.upper
    (None, None)

    Both upper and lower bounds can be specified:

    >>> idx = Idx(i, (L, U)); idx
    i
    >>> idx.lower, idx.upper
    (L, U)

    When only a single bound is given it is interpreted as the dimension
    and the lower bound defaults to 0:

    >>> idx = Idx(i, n); idx.lower, idx.upper
    (0, n - 1)
    >>> idx = Idx(i, 4); idx.lower, idx.upper
    (0, 3)
    >>> idx = Idx(i, oo); idx.lower, idx.upper
    (0, oo)

    """
    is_integer = True
    is_finite = True
    is_real = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True

    def __new__(cls, label, range=None, **kw_args):
        if isinstance(label, str):
            label = Symbol(label, integer=True)
        label, range = list(map(sympify, (label, range)))
        if label.is_Number:
            if not label.is_integer:
                raise TypeError('Index is not an integer number.')
            return label
        if not label.is_integer:
            raise TypeError('Idx object requires an integer label.')
        elif is_sequence(range):
            if len(range) != 2:
                raise ValueError(filldedent('\n                    Idx range tuple must have length 2, but got %s' % len(range)))
            for bound in range:
                if bound.is_integer is False and bound is not S.Infinity and (bound is not S.NegativeInfinity):
                    raise TypeError('Idx object requires integer bounds.')
            args = (label, Tuple(*range))
        elif isinstance(range, Expr):
            if range is not S.Infinity and fuzzy_not(range.is_integer):
                raise TypeError('Idx object requires an integer dimension.')
            args = (label, Tuple(0, range - 1))
        elif range:
            raise TypeError(filldedent('\n                The range must be an ordered iterable or\n                integer SymPy expression.'))
        else:
            args = (label,)
        obj = Expr.__new__(cls, *args, **kw_args)
        obj._assumptions['finite'] = True
        obj._assumptions['real'] = True
        return obj

    @property
    def label(self):
        """Returns the label (Integer or integer expression) of the Idx object.

        Examples
        ========

        >>> from sympy import Idx, Symbol
        >>> x = Symbol('x', integer=True)
        >>> Idx(x).label
        x
        >>> j = Symbol('j', integer=True)
        >>> Idx(j).label
        j
        >>> Idx(j + 1).label
        j + 1

        """
        return self.args[0]

    @property
    def lower(self):
        """Returns the lower bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).lower
        0
        >>> Idx('j', 5).lower
        0
        >>> Idx('j').lower is None
        True

        """
        try:
            return self.args[1][0]
        except IndexError:
            return

    @property
    def upper(self):
        """Returns the upper bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).upper
        1
        >>> Idx('j', 5).upper
        4
        >>> Idx('j').upper is None
        True

        """
        try:
            return self.args[1][1]
        except IndexError:
            return

    def _sympystr(self, p):
        return p.doprint(self.label)

    @property
    def name(self):
        return self.label.name if self.label.is_Symbol else str(self.label)

    @property
    def free_symbols(self):
        return {self}