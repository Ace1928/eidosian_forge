from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class
class AppliedPredicate(Boolean):
    """
    The class of expressions resulting from applying ``Predicate`` to
    the arguments. ``AppliedPredicate`` merely wraps its argument and
    remain unevaluated. To evaluate it, use the ``ask()`` function.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> Q.integer(1)
    Q.integer(1)

    The ``function`` attribute returns the predicate, and the ``arguments``
    attribute returns the tuple of arguments.

    >>> type(Q.integer(1))
    <class 'sympy.assumptions.assume.AppliedPredicate'>
    >>> Q.integer(1).function
    Q.integer
    >>> Q.integer(1).arguments
    (1,)

    Applied predicates can be evaluated to a boolean value with ``ask``:

    >>> ask(Q.integer(1))
    True

    """
    __slots__ = ()

    def __new__(cls, predicate, *args):
        if not isinstance(predicate, Predicate):
            raise TypeError('%s is not a Predicate.' % predicate)
        args = map(_sympify, args)
        return super().__new__(cls, predicate, *args)

    @property
    def arg(self):
        """
        Return the expression used by this assumption.

        Examples
        ========

        >>> from sympy import Q, Symbol
        >>> x = Symbol('x')
        >>> a = Q.integer(x + 1)
        >>> a.arg
        x + 1

        """
        args = self._args
        if len(args) == 2:
            return args[1]
        raise TypeError("'arg' property is allowed only for unary predicates.")

    @property
    def function(self):
        """
        Return the predicate.
        """
        return self._args[0]

    @property
    def arguments(self):
        """
        Return the arguments which are applied to the predicate.
        """
        return self._args[1:]

    def _eval_ask(self, assumptions):
        return self.function.eval(self.arguments, assumptions)

    @property
    def binary_symbols(self):
        from .ask import Q
        if self.function == Q.is_true:
            i = self.arguments[0]
            if i.is_Boolean or i.is_Symbol:
                return i.binary_symbols
        if self.function in (Q.eq, Q.ne):
            if true in self.arguments or false in self.arguments:
                if self.arguments[0].is_Symbol:
                    return {self.arguments[0]}
                elif self.arguments[1].is_Symbol:
                    return {self.arguments[1]}
        return set()