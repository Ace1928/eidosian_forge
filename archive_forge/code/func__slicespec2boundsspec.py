import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
@staticmethod
def _slicespec2boundsspec(slicespec, scs):
    """
        Convert an iterable slicespec (supplying r1,r2,c1,c2 of a
        Slice) into a BoundingRegion specification.

        Exact inverse of _boundsspec2slicespec().
        """
    r1, r2, c1, c2 = slicespec
    left, bottom = scs.matrix2sheet(r2, c1)
    right, top = scs.matrix2sheet(r1, c2)
    return ((left, bottom), (right, top))