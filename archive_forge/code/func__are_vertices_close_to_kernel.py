from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _are_vertices_close_to_kernel(verts, snappeaVerts):
    for key, vert in verts.items():
        snappeaVert = snappeaVerts[key]
        if not _is_vertex_close_to_kernel(vert, snappeaVert):
            return False
    return True