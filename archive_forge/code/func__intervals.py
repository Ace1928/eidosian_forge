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
def _intervals(self, sym, err_on_Eq=False):
    """Return a bool and a message (when bool is False), else a
        list of unique tuples, (a, b, e, i), where a and b
        are the lower and upper bounds in which the expression e of
        argument i in self is defined and $a < b$ (when involving
        numbers) or $a \\le b$ when involving symbols.

        If there are any relationals not involving sym, or any
        relational cannot be solved for sym, the bool will be False
        a message be given as the second return value. The calling
        routine should have removed such relationals before calling
        this routine.

        The evaluated conditions will be returned as ranges.
        Discontinuous ranges will be returned separately with
        identical expressions. The first condition that evaluates to
        True will be returned as the last tuple with a, b = -oo, oo.
        """
    from sympy.solvers.inequalities import _solve_inequality
    assert isinstance(self, Piecewise)

    def nonsymfail(cond):
        return (False, filldedent('\n                A condition not involving\n                %s appeared: %s' % (sym, cond)))

    def _solve_relational(r):
        if sym not in r.free_symbols:
            return nonsymfail(r)
        try:
            rv = _solve_inequality(r, sym)
        except NotImplementedError:
            return (False, 'Unable to solve relational %s for %s.' % (r, sym))
        if isinstance(rv, Relational):
            free = rv.args[1].free_symbols
            if rv.args[0] != sym or sym in free:
                return (False, 'Unable to solve relational %s for %s.' % (r, sym))
            if rv.rel_op == '==':
                rv = S.false
            elif rv.rel_op == '!=':
                try:
                    rv = Or(sym < rv.rhs, sym > rv.rhs)
                except TypeError:
                    rv = S.true
        elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
            rv = S.true
        return (True, rv)
    args = list(self.args)
    keys = self.atoms(Relational)
    reps = {}
    for r in keys:
        ok, s = _solve_relational(r)
        if ok != True:
            return (False, ok)
        reps[r] = s
    args = [i.xreplace(reps) for i in self.args]
    expr_cond = []
    default = idefault = None
    for i, (expr, cond) in enumerate(args):
        if cond is S.false:
            continue
        if cond is S.true:
            default = expr
            idefault = i
            break
        if isinstance(cond, Eq):
            if err_on_Eq:
                return (False, 'encountered Eq condition: %s' % cond)
            continue
        cond = to_cnf(cond)
        if isinstance(cond, And):
            cond = distribute_or_over_and(cond)
        if isinstance(cond, Or):
            expr_cond.extend([(i, expr, o) for o in cond.args if not isinstance(o, Eq)])
        elif cond is not S.false:
            expr_cond.append((i, expr, cond))
        elif cond is S.true:
            default = expr
            idefault = i
            break
    int_expr = []
    for iarg, expr, cond in expr_cond:
        if isinstance(cond, And):
            lower = S.NegativeInfinity
            upper = S.Infinity
            exclude = []
            for cond2 in cond.args:
                if not isinstance(cond2, Relational):
                    return (False, 'expecting only Relationals')
                if isinstance(cond2, Eq):
                    lower = upper
                    if err_on_Eq:
                        return (False, 'encountered secondary Eq condition')
                    break
                elif isinstance(cond2, Ne):
                    l, r = cond2.args
                    if l == sym:
                        exclude.append(r)
                    elif r == sym:
                        exclude.append(l)
                    else:
                        return nonsymfail(cond2)
                    continue
                elif cond2.lts == sym:
                    upper = Min(cond2.gts, upper)
                elif cond2.gts == sym:
                    lower = Max(cond2.lts, lower)
                else:
                    return nonsymfail(cond2)
            if exclude:
                exclude = list(ordered(exclude))
                newcond = []
                for i, e in enumerate(exclude):
                    if e < lower == True or e > upper == True:
                        continue
                    if not newcond:
                        newcond.append((None, lower))
                    newcond.append((newcond[-1][1], e))
                newcond.append((newcond[-1][1], upper))
                newcond.pop(0)
                expr_cond.extend([(iarg, expr, And(i[0] < sym, sym < i[1])) for i in newcond])
                continue
        elif isinstance(cond, Relational) and cond.rel_op != '!=':
            lower, upper = (cond.lts, cond.gts)
            if cond.lts == sym:
                lower = S.NegativeInfinity
            elif cond.gts == sym:
                upper = S.Infinity
            else:
                return nonsymfail(cond)
        else:
            return (False, 'unrecognized condition: %s' % cond)
        lower, upper = (lower, Max(lower, upper))
        if err_on_Eq and lower == upper:
            return (False, 'encountered Eq condition')
        if (lower >= upper) is not S.true:
            int_expr.append((lower, upper, expr, iarg))
    if default is not None:
        int_expr.append((S.NegativeInfinity, S.Infinity, default, idefault))
    return (True, list(uniq(int_expr)))