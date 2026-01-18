from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
def _logaddexp2(x1, x2, *, evaluate=True):
    return _lb(Add(_exp2(x1, evaluate=evaluate), _exp2(x2, evaluate=evaluate), evaluate=evaluate))