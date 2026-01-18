from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
class adjoint(Function):
    """
    Conjugate transpose or Hermite conjugation.

    Examples
    ========

    >>> from sympy import adjoint, MatrixSymbol
    >>> A = MatrixSymbol('A', 10, 5)
    >>> adjoint(A)
    Adjoint(A)

    Parameters
    ==========

    arg : Matrix
        Matrix or matrix expression to take the adjoint of.

    Returns
    =======

    value : Matrix
        Represents the conjugate transpose or Hermite
        conjugation of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_adjoint()
        if obj is not None:
            return obj
        obj = arg._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def _eval_adjoint(self):
        return self.args[0]

    def _eval_conjugate(self):
        return transpose(self.args[0])

    def _eval_transpose(self):
        return conjugate(self.args[0])

    def _latex(self, printer, exp=None, *args):
        arg = printer._print(self.args[0])
        tex = '%s^{\\dagger}' % arg
        if exp:
            tex = '\\left(%s\\right)^{%s}' % (tex, exp)
        return tex

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if printer._use_unicode:
            pform = pform ** prettyForm('†')
        else:
            pform = pform ** prettyForm('+')
        return pform