from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def acceptable_error(poly, z, a, portion_bad):
    """
    A error is judged as acceptable if poly(z) = a to within
    2^-(portion_bad*z.prec())
    """
    return error(poly, z, a) <= floor(portion_bad * z.prec())