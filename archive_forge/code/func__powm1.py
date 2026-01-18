from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin
def _powm1(x, y, *, evaluate=True):
    return Add(Pow(x, y, evaluate=evaluate), -S.One, evaluate=evaluate)