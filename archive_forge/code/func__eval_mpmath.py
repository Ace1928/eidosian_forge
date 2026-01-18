from sympy.core import S
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
def _eval_mpmath(self):
    return (betainc_mpmath_fix, (*self.args, S(1)))