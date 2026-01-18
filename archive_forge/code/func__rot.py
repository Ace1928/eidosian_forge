from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _rot(self, axis, angle):
    """DCM for simple axis 1,2,or 3 rotations."""
    if axis == 1:
        return Matrix([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])
    elif axis == 2:
        return Matrix([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
    elif axis == 3:
        return Matrix([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])