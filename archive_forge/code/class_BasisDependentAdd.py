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
class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    """

    def __new__(cls, *args, **options):
        components = {}
        for i, arg in enumerate(args):
            if not isinstance(arg, cls._expr_type):
                if isinstance(arg, Mul):
                    arg = cls._mul_func(*arg.args)
                elif isinstance(arg, Add):
                    arg = cls._add_func(*arg.args)
                else:
                    raise TypeError(str(arg) + ' cannot be interpreted correctly')
            if arg == cls.zero:
                continue
            if hasattr(arg, 'components'):
                for x in arg.components:
                    components[x] = components.get(x, 0) + arg.components[x]
        temp = list(components.keys())
        for x in temp:
            if components[x] == 0:
                del components[x]
        if len(components) == 0:
            return cls.zero
        newargs = [x * components[x] for x in components]
        obj = super().__new__(cls, *newargs, **options)
        if isinstance(obj, Mul):
            return cls._mul_func(*obj.args)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = components
        obj._sys = list(components.keys())[0]._sys
        return obj