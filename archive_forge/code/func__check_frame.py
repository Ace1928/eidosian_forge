from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _check_frame(other):
    from .vector import VectorTypeError
    if not isinstance(other, ReferenceFrame):
        raise VectorTypeError(other, ReferenceFrame('A'))