from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x
class my_function(Function):

    def fdiff(self, argindex=1):
        return cos(self.args[0])

    @classmethod
    def eval(cls, arg):
        arg = sympify(arg)
        if arg == 0:
            return sympify(0)