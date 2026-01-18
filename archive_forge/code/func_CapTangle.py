import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def CapTangle():
    """The unknotted (2,0) tangle."""
    cap = Strand('cap')
    return Tangle((2, 0), [cap], [(cap, 0), (cap, 1)])