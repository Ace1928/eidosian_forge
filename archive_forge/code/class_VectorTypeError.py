from sympy.core.backend import (S, sympify, expand, sqrt, Add, zeros, acos,
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin
from mpmath.libmp.libmpf import prec_to_dps
class VectorTypeError(TypeError):

    def __init__(self, other, want):
        msg = filldedent("Expected an instance of %s, but received object '%s' of %s." % (type(want), other, type(other)))
        super().__init__(msg)