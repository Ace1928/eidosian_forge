from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def _crawl(expr, func, *args, **kwargs):
    """Crawl the expression tree, and apply func to every node."""
    val = func(expr, *args, **kwargs)
    if val is not None:
        return val
    new_args = (_crawl(arg, func, *args, **kwargs) for arg in expr.args)
    return expr.func(*new_args)