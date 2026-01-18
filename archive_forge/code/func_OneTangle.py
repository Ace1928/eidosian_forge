import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def OneTangle():
    """The one tangle, equivalent to ``RationalTangle(1)``."""
    c = Crossing('one')
    return Tangle(2, [c], [(c, 0), (c, 1), (c, 3), (c, 2)], 'OneTangle')