from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _compute_translations(vertex):
    vertex.Translations = [ComplexCuspCrossSection._get_translation(vertex, i) for i in range(2)]