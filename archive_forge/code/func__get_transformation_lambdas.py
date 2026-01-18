from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrices import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
from sympy.vector.vector import BaseVector
@staticmethod
def _get_transformation_lambdas(curv_coord_name):
    """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
    if isinstance(curv_coord_name, str):
        if curv_coord_name == 'cartesian':
            return lambda x, y, z: (x, y, z)
        if curv_coord_name == 'spherical':
            return lambda r, theta, phi: (r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta))
        if curv_coord_name == 'cylindrical':
            return lambda r, theta, h: (r * cos(theta), r * sin(theta), h)
        raise ValueError('Wrong set of parameters.Type of coordinate system is defined')