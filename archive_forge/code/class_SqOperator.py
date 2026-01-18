from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class SqOperator(Expr):
    """
    Base class for Second Quantization operators.
    """
    op_symbol = 'sq'
    is_commutative = False

    def __new__(cls, k):
        obj = Basic.__new__(cls, sympify(k))
        return obj

    @property
    def state(self):
        """
        Returns the state index related to this operator.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F, Fd, B, Bd
        >>> p = Symbol('p')
        >>> F(p).state
        p
        >>> Fd(p).state
        p
        >>> B(p).state
        p
        >>> Bd(p).state
        p

        """
        return self.args[0]

    @property
    def is_symbolic(self):
        """
        Returns True if the state is a symbol (as opposed to a number).

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> p = Symbol('p')
        >>> F(p).is_symbolic
        True
        >>> F(1).is_symbolic
        False

        """
        if self.state.is_Integer:
            return False
        else:
            return True

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return '%s(%r)' % (self.op_symbol, self.state)

    def apply_operator(self, state):
        """
        Applies an operator to itself.
        """
        raise NotImplementedError('implement apply_operator in a subclass')