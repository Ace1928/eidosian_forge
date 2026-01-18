from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _sort_args(self, args):
    lists, atoms = sift(args, lambda a: isinstance(a, (tuple, list)), binary=True)
    return (atoms, lists)