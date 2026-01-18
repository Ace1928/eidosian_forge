from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def _handle_irel(self, x, handler):
    """Return either None (if the conditions of self depend only on x) else
        a Piecewise expression whose expressions (handled by the handler that
        was passed) are paired with the governing x-independent relationals,
        e.g. Piecewise((A, a(x) & b(y)), (B, c(x) | c(y)) ->
        Piecewise(
            (handler(Piecewise((A, a(x) & True), (B, c(x) | True)), b(y) & c(y)),
            (handler(Piecewise((A, a(x) & True), (B, c(x) | False)), b(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | True)), c(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | False)), True))
        """
    rel = self.atoms(Relational)
    irel = list(ordered([r for r in rel if x not in r.free_symbols and r not in (S.true, S.false)]))
    if irel:
        args = {}
        exprinorder = []
        for truth in product((1, 0), repeat=len(irel)):
            reps = dict(zip(irel, truth))
            if 1 not in truth:
                cond = None
            else:
                andargs = Tuple(*[i for i in reps if reps[i]])
                free = list(andargs.free_symbols)
                if len(free) == 1:
                    from sympy.solvers.inequalities import reduce_inequalities, _solve_inequality
                    try:
                        t = reduce_inequalities(andargs, free[0])
                    except (ValueError, NotImplementedError):
                        t = And(*[_solve_inequality(a, free[0], linear=True) for a in andargs])
                else:
                    t = And(*andargs)
                if t is S.false:
                    continue
                cond = t
            expr = handler(self.xreplace(reps))
            if isinstance(expr, self.func) and len(expr.args) == 1:
                expr, econd = expr.args[0]
                cond = And(econd, True if cond is None else cond)
            if cond is not None:
                args.setdefault(expr, []).append(cond)
                exprinorder.append(expr)
        for k in args:
            args[k] = Or(*args[k])
        args = [(e, args[e]) for e in uniq(exprinorder)]
        args.append((expr, True))
        return Piecewise(*args)