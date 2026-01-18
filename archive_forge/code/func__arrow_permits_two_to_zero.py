from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def _arrow_permits_two_to_zero(self, arrow):
    edge = arrow.axis()
    if not edge.IntOrBdry == 'int':
        return (False, 'Cannot do move on exterior edge')
    if edge.valence() != 2:
        return (False, 'Edge has valence %d not 2' % edge.valence())
    if not edge.distinct():
        return (False, 'Tets around edge are not distinct')
    if arrow.equator() == arrow.glued().equator():
        return (False, 'Edges opposite the valence 2 edge are the same')
    return (True, None)