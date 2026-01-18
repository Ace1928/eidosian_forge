from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _get_normalized_translations(vertex):
    """
        Compute the translations corresponding to the merdian and longitude of
        the given cusp.
        """
    m, l = vertex.Translations
    return (m / l * abs(l), abs(l))