from __future__ import annotations
from typing import TYPE_CHECKING
from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr
class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    """

    def __new__(cls, *args, **options):
        from sympy.vector import Cross, Dot, Curl, Gradient
        count = 0
        measure_number = S.One
        zeroflag = False
        extra_args = []
        for arg in args:
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            elif arg == S.Zero:
                zeroflag = True
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            else:
                measure_number *= arg
        if count > 1:
            raise ValueError('Invalid multiplication')
        elif count == 0:
            return Mul(*args, **options)
        if zeroflag:
            return cls.zero
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for x in expr.args]
            return cls._add_func(*newargs)
        obj = super().__new__(cls, measure_number, expr._base_instance, *extra_args, **options)
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = {expr._base_instance: measure_number}
        obj._sys = expr._base_instance._sys
        return obj

    def _sympystr(self, printer):
        measure_str = printer._print(self._measure_number)
        if '(' in measure_str or '-' in measure_str or '+' in measure_str:
            measure_str = '(' + measure_str + ')'
        return measure_str + '*' + printer._print(self._base_instance)