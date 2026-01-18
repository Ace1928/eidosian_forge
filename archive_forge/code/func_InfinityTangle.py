import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def InfinityTangle():
    """The infinity tangle, equivalent to ``RationalTangle(1, 0)`` or
    ``IdentityBraid(2)``."""
    left, right = (Strand('L'), Strand('R'))
    return Tangle(2, [left, right], [(left, 0), (right, 0), (left, 1), (right, 1)], 'InfinityTangle')