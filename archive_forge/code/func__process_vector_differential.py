from functools import reduce
from sympy.core.backend import (sympify, diff, sin, cos, Matrix, symbols,
from sympy.integrals.integrals import integrate
from sympy.simplify.trigsimp import trigsimp
from .vector import Vector, _check_vector
from .frame import CoordinateSym, _check_frame
from .dyadic import Dyadic
from .printing import vprint, vsprint, vpprint, vlatex, init_vprinting
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import translate
def _process_vector_differential(vectdiff, condition, variable, ordinate, frame):
    """
        Helper function for get_motion methods. Finds derivative of vectdiff
        wrt variable, and its integral using the specified boundary condition
        at value of variable = ordinate.
        Returns a tuple of - (derivative, function and integral) wrt vectdiff

        """
    if condition != 0:
        condition = express(condition, frame, variables=True)
    if vectdiff == Vector(0):
        return (0, 0, condition)
    vectdiff1 = express(vectdiff, frame)
    vectdiff2 = time_derivative(vectdiff, frame)
    vectdiff0 = Vector(0)
    lims = (variable, ordinate, variable)
    for dim in frame:
        function1 = vectdiff1.dot(dim)
        abscissa = dim.dot(condition).subs({variable: ordinate})
        vectdiff0 += (integrate(function1, lims) + abscissa) * dim
    return (vectdiff2, vectdiff, vectdiff0)