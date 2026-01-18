from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def _recurser(expr, sub_dict):
    num, den = _fraction_decomp(expr)
    if den != 1:
        denom_subbed = _recurser(den, sub_dict)
        if denom_subbed.evalf() == 0:
            expr = simplify(expr)
        else:
            num_subbed = _recurser(num, sub_dict)
            return num_subbed / denom_subbed
    val = _sub_func(expr, sub_dict)
    if val is not None:
        return val
    new_args = (_recurser(arg, sub_dict) for arg in expr.args)
    return expr.func(*new_args)