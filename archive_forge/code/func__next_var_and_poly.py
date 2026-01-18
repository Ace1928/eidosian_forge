from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def _next_var_and_poly(polys, extension_vars):
    """
    Applies _only_var_left_in_poly to find a polynomial that has
    one free variable and returns pair (variable, polynomial).
    """
    for poly in polys:
        var = _only_var_left_in_poly(poly, extension_vars)
        if var:
            return (poly, var)
    raise Exception('Could not find polynomial becoming univariate after substituition, the Groebner basis you are tryin to solve is probably not in lexicographic order or of a 0-dimensional ideal!')