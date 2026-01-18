from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def add_boundary_slope(surface, cusp_equations):
    surface.BoundarySlope = (-dot_product(surface.Shifts, cusp_equations[1]), dot_product(surface.Shifts, cusp_equations[0]))