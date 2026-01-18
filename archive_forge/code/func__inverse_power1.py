import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def _inverse_power1(zl, zu, yl, yu, orig_xl, orig_xu, feasibility_tol):
    """z = x**y => compute bounds on x.

    First, start by computing bounds on x with

        x = exp(ln(z) / y)

    However, if y is an integer, then x can be negative, so there are
    several special cases. See the docs below.

    """
    xl, xu = log(zl, zu)
    xl, xu = div(xl, xu, yl, yu, feasibility_tol)
    xl, xu = exp(xl, xu)
    if yl == yu and yl == round(yl):
        y = yl
        if y == 0:
            xl = -inf
            xu = inf
        elif y % 2 == 0:
            'if y is even, then there are two primary cases (note that it is much\n            easier to walk through these while looking at plots):\n\n            case 1: y is positive\n\n                x**y is convex, positive, and symmetric. The bounds on x\n                depend on the lower bound of z. If zl <= 0, then xl\n                should simply be -xu. However, if zl > 0, then we may be\n                able to say something better. For example, if the\n                original lower bound on x is positive, then we can keep\n                xl computed from x = exp(ln(z) / y). Furthermore, if the\n                original lower bound on x is larger than -xl computed\n                from x = exp(ln(z) / y), then we can still keep the xl\n                computed from x = exp(ln(z) / y). Similar logic applies\n                to the upper bound of x.\n\n            case 2: y is negative\n\n                The ideas are similar to case 1.\n\n            '
            if zu + feasibility_tol < 0:
                raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
            if y > 0:
                if zu <= 0:
                    _xl = 0
                    _xu = 0
                elif zl <= 0:
                    _xl = -xu
                    _xu = xu
                else:
                    if orig_xl <= -xl + feasibility_tol:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl - feasibility_tol:
                        _xu = -xl
                    else:
                        _xu = xu
                xl = _xl
                xu = _xu
            else:
                if zu == 0:
                    raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
                elif zl <= 0:
                    _xl = -inf
                    _xu = inf
                else:
                    if orig_xl <= -xl + feasibility_tol:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl - feasibility_tol:
                        _xu = -xl
                    else:
                        _xu = xu
                xl = _xl
                xu = _xu
        else:
            'y is odd.\n\n            Case 1: y is positive\n\n                x**y is monotonically increasing. If y is positive, then\n                we can can compute the bounds on x using x = z**(1/y)\n                and the signs on xl and xu depend on the signs of zl and\n                zu.\n\n            Case 2: y is negative\n\n                Again, this is easier to visualize with a plot. x**y\n                approaches zero when x approaches -inf or inf.  Thus, if\n                zl < 0 < zu, then no bounds can be inferred for x. If z\n                is positive (zl >=0 ) then we can use the bounds\n                computed from x = exp(ln(z) / y). If z is negative (zu\n                <= 0), then we live in the bottom left quadrant, xl\n                depends on zu, and xu depends on zl.\n\n            '
            if y > 0:
                xl = abs(zl) ** (1.0 / y)
                xl = math.copysign(xl, zl)
                xu = abs(zu) ** (1.0 / y)
                xu = math.copysign(xu, zu)
            elif zl >= 0:
                pass
            elif zu <= 0:
                if zu == 0:
                    xl = -inf
                else:
                    xl = -abs(zu) ** (1.0 / y)
                if zl == 0:
                    xu = -inf
                else:
                    xu = -abs(zl) ** (1.0 / y)
            else:
                xl = -inf
                xu = inf
    return (xl, xu)