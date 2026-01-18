from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
class IncompleteCuspError(RuntimeError):
    """
    Exception raised when trying to construct a CuspCrossSection
    from a Manifold with Dehn-fillings.
    """

    def __init__(self, manifold):
        self.manifold = manifold

    def __str__(self):
        return 'Cannot construct CuspCrossSection from manifold with Dehn-fillings: %s' % self.manifold