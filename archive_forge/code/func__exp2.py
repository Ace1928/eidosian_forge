from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
def _exp2(x, *, evaluate=True):
    return Pow(_two, x, evaluate=evaluate)