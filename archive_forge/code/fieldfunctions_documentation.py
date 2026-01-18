from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector

    Returns the scalar potential difference between two points in a
    certain frame, wrt a given field.

    If a scalar field is provided, its values at the two points are
    considered. If a conservative vector field is provided, the values
    of its scalar potential function at the two points are used.

    Returns (potential at position 2) - (potential at position 1)

    Parameters
    ==========

    field : Vector/sympyfiable
        The field to calculate wrt

    frame : ReferenceFrame
        The frame to do the calculations in

    point1 : Point
        The initial Point in given frame

    position2 : Point
        The second Point in the given frame

    origin : Point
        The Point to use as reference point for position vector
        calculation

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, Point
    >>> from sympy.physics.vector import scalar_potential_difference
    >>> R = ReferenceFrame('R')
    >>> O = Point('O')
    >>> P = O.locatenew('P', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    >>> vectfield = 4*R[0]*R[1]*R.x + 2*R[0]**2*R.y
    >>> scalar_potential_difference(vectfield, R, O, P, O)
    2*R_x**2*R_y
    >>> Q = O.locatenew('O', 3*R.x + R.y + 2*R.z)
    >>> scalar_potential_difference(vectfield, R, P, Q, O)
    -2*R_x**2*R_y + 18

    