from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _is_vertex_close_to_kernel(vertex, snappeaVertex):
    if vertex == Infinity or snappeaVertex == Infinity:
        return vertex == snappeaVertex
    return _is_number_close_to_kernel(vertex, snappeaVertex)