from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def _higher_order_ode_solver(match):
    eqs = match['eq']
    funcs = match['func']
    t = match['t']
    sysorder = match['order']
    type = match.get('type_of_equation', 'type0')
    is_second_order = match.get('is_second_order', False)
    is_transformed = match.get('is_transformed', False)
    is_euler = is_transformed and type == 'type1'
    is_higher_order_type2 = is_transformed and type == 'type2' and ('P' in match)
    if is_second_order:
        new_eqs, new_funcs = _second_order_to_first_order(eqs, funcs, t, A1=match.get('A1', None), A0=match.get('A0', None), b=match.get('rhs', None), type=type, t_=match.get('t_', None))
    else:
        new_eqs, new_funcs = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs, type=type, J=match.get('J', None), f_t=match.get('f(t)', None), P=match.get('P', None), b=match.get('rhs', None))
    if is_transformed:
        t = match.get('t_', t)
    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])
    sol = None
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None
    if sol is None:
        return sol
    is_second_order_type2 = is_second_order and type == 'type2'
    underscores = '__' if is_transformed else '_'
    sol = _select_equations(sol, funcs, key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))
    if match.get('is_transformed', False):
        if is_second_order_type2:
            g_t = match['g(t)']
            tau = match['tau']
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        elif is_euler:
            t = match['t']
            tau = match['t_']
            sol = [s.subs(tau, log(t)) for s in sol]
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    return sol