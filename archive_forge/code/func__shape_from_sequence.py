import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _shape_from_sequence(footprints, require_odd_size=False):
    """Determine the shape of composite footprint

    In the future if we only want to support odd-sized square, we may want to
    change this to require_odd_size
    """
    if not _footprint_is_sequence(footprints):
        raise ValueError('expected a sequence of footprints')
    ndim = footprints[0][0].ndim
    shape = [0] * ndim

    def _odd_size(size, require_odd_size):
        if require_odd_size and size % 2 == 0:
            raise ValueError('expected all footprint elements to have odd size')
    for d in range(ndim):
        fp, nreps = footprints[0]
        _odd_size(fp.shape[d], require_odd_size)
        shape[d] = fp.shape[d] + (nreps - 1) * (fp.shape[d] - 1)
        for fp, nreps in footprints[1:]:
            _odd_size(fp.shape[d], require_odd_size)
            shape[d] += nreps * (fp.shape[d] - 1)
    return tuple(shape)