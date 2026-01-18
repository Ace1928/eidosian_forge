from sympy.core import Mul
from sympy.core.function import count_ops
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions import gamma
from sympy.simplify.gammasimp import gammasimp, _gammasimp
from sympy.utilities.timeutils import timethis
def _gamma_as_comb(expr):
    """
    Helper function for combsimp.

    Rewrites expression in terms of factorials and binomials
    """
    expr = expr.rewrite(factorial)

    def f(rv):
        if not rv.is_Mul:
            return rv
        rvd = rv.as_powers_dict()
        nd_fact_args = [[], []]
        for k in rvd:
            if isinstance(k, factorial) and rvd[k].is_Integer:
                if rvd[k].is_positive:
                    nd_fact_args[0].extend([k.args[0]] * rvd[k])
                else:
                    nd_fact_args[1].extend([k.args[0]] * -rvd[k])
                rvd[k] = 0
        if not nd_fact_args[0] or not nd_fact_args[1]:
            return rv
        hit = False
        for m in range(2):
            i = 0
            while i < len(nd_fact_args[m]):
                ai = nd_fact_args[m][i]
                for j in range(i + 1, len(nd_fact_args[m])):
                    aj = nd_fact_args[m][j]
                    sum = ai + aj
                    if sum in nd_fact_args[1 - m]:
                        hit = True
                        nd_fact_args[1 - m].remove(sum)
                        del nd_fact_args[m][j]
                        del nd_fact_args[m][i]
                        rvd[binomial(sum, ai if count_ops(ai) < count_ops(aj) else aj)] += -1 if m == 0 else 1
                        break
                else:
                    i += 1
        if hit:
            return Mul(*[k ** rvd[k] for k in rvd] + [factorial(k) for k in nd_fact_args[0]]) / Mul(*[factorial(k) for k in nd_fact_args[1]])
        return rv
    return bottom_up(expr, f)