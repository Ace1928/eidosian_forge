from sympy.core.logic import fuzzy_and
from sympy.simplify.simplify import nsimplify
from .interval_membership import intervalMembership
def _pow_int(inter, power):
    """Evaluates an interval raised to an integer power"""
    power = int(power)
    if power & 1:
        return interval(inter.start ** power, inter.end ** power)
    elif inter.start < 0 and inter.end > 0:
        start = 0
        end = max(inter.start ** power, inter.end ** power)
        return interval(start, end)
    else:
        return interval(inter.start ** power, inter.end ** power)