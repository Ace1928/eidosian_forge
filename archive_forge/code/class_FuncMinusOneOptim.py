from sympy.core.function import expand_log
from sympy.core.singleton import S
from sympy.core.symbol import Wild
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.assumptions import Q, ask
from sympy.codegen.cfunctions import log1p, log2, exp2, expm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.expr import UnevaluatedExpr
from sympy.core.power import Pow
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.core.mul import Mul
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.utilities.iterables import sift
class FuncMinusOneOptim(ReplaceOptim):
    """Specialization of ReplaceOptim for functions evaluating "f(x) - 1".

    Explanation
    ===========

    Numerical functions which go toward one as x go toward zero is often best
    implemented by a dedicated function in order to avoid catastrophic
    cancellation. One such example is ``expm1(x)`` in the C standard library
    which evaluates ``exp(x) - 1``. Such functions preserves many more
    significant digits when its argument is much smaller than one, compared
    to subtracting one afterwards.

    Parameters
    ==========

    func :
        The function which is subtracted by one.
    func_m_1 :
        The specialized function evaluating ``func(x) - 1``.
    opportunistic : bool
        When ``True``, apply the transformation as long as the magnitude of the
        remaining number terms decreases. When ``False``, only apply the
        transformation if it completely eliminates the number term.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.codegen.rewriting import FuncMinusOneOptim
    >>> from sympy.codegen.cfunctions import expm1
    >>> x, y = symbols('x y')
    >>> expm1_opt = FuncMinusOneOptim(exp, expm1)
    >>> expm1_opt(exp(x) + 2*exp(5*y) - 3)
    expm1(x) + 2*expm1(5*y)


    """

    def __init__(self, func, func_m_1, opportunistic=True):
        weight = 10
        super().__init__(lambda e: e.is_Add, self.replace_in_Add, cost_function=lambda expr: expr.count_ops() - weight * expr.count(func_m_1))
        self.func = func
        self.func_m_1 = func_m_1
        self.opportunistic = opportunistic

    def _group_Add_terms(self, add):
        numbers, non_num = sift(add.args, lambda arg: arg.is_number, binary=True)
        numsum = sum(numbers)
        terms_with_func, other = sift(non_num, lambda arg: arg.has(self.func), binary=True)
        return (numsum, terms_with_func, other)

    def replace_in_Add(self, e):
        """ passed as second argument to Basic.replace(...) """
        numsum, terms_with_func, other_non_num_terms = self._group_Add_terms(e)
        if numsum == 0:
            return e
        substituted, untouched = ([], [])
        for with_func in terms_with_func:
            if with_func.is_Mul:
                func, coeff = sift(with_func.args, lambda arg: arg.func == self.func, binary=True)
                if len(func) == 1 and len(coeff) == 1:
                    func, coeff = (func[0], coeff[0])
                else:
                    coeff = None
            elif with_func.func == self.func:
                func, coeff = (with_func, S.One)
            else:
                coeff = None
            if coeff is not None and coeff.is_number and (sign(coeff) == -sign(numsum)):
                if self.opportunistic:
                    do_substitute = abs(coeff + numsum) < abs(numsum)
                else:
                    do_substitute = coeff + numsum == 0
                if do_substitute:
                    numsum += coeff
                    substituted.append(coeff * self.func_m_1(*func.args))
                    continue
            untouched.append(with_func)
        return e.func(numsum, *substituted, *untouched, *other_non_num_terms)

    def __call__(self, expr):
        alt1 = super().__call__(expr)
        alt2 = super().__call__(expr.factor())
        return self.cheapest(alt1, alt2)