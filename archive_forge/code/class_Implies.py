from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
class Implies(BooleanFunction):
    """
    Logical implication.

    A implies B is equivalent to if A then B. Mathematically, it is written
    as `A \\Rightarrow B` and is equivalent to `\\neg A \\vee B` or ``~A | B``.

    Accepts two Boolean arguments; A and B.
    Returns False if A is True and B is False
    Returns True otherwise.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Implies(True, False)
    False
    >>> Implies(False, False)
    True
    >>> Implies(True, True)
    True
    >>> Implies(False, True)
    True
    >>> x >> y
    Implies(x, y)
    >>> y << x
    Implies(x, y)

    Notes
    =====

    The ``>>`` and ``<<`` operators are provided as a convenience, but note
    that their use here is different from their normal use in Python, which is
    bit shifts. Hence, ``Implies(a, b)`` and ``a >> b`` will return different
    things if ``a`` and ``b`` are integers.  In particular, since Python
    considers ``True`` and ``False`` to be integers, ``True >> True`` will be
    the same as ``1 >> 1``, i.e., 0, which has a truth value of False.  To
    avoid this issue, use the SymPy objects ``true`` and ``false``.

    >>> from sympy import true, false
    >>> True >> False
    1
    >>> true >> false
    False

    """

    @classmethod
    def eval(cls, *args):
        try:
            newargs = []
            for x in args:
                if isinstance(x, Number) or x in (0, 1):
                    newargs.append(bool(x))
                else:
                    newargs.append(x)
            A, B = newargs
        except ValueError:
            raise ValueError('%d operand(s) used for an Implies (pairs are required): %s' % (len(args), str(args)))
        if A in (True, False) or B in (True, False):
            return Or(Not(A), B)
        elif A == B:
            return true
        elif A.is_Relational and B.is_Relational:
            if A.canonical == B.canonical:
                return true
            if A.negated.canonical == B.canonical:
                return B
        else:
            return Basic.__new__(cls, *args)

    def to_nnf(self, simplify=True):
        a, b = self.args
        return Or._to_nnf(Not(a), b, simplify=simplify)

    def to_anf(self, deep=True):
        a, b = self.args
        return Xor._to_anf(true, a, And(a, b), deep=deep)