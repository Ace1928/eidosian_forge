import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
class BboxTransformToMaxOnly(BboxTransformTo):
    """
    `BboxTransformTo` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox` with a fixed upper left of (0, 0).
    """

    def get_matrix(self):
        if self._invalid:
            xmax, ymax = self._boxout.max
            if DEBUG and (xmax == 0 or ymax == 0):
                raise ValueError('Transforming to a singular bounding box.')
            self._mtx = np.array([[xmax, 0.0, 0.0], [0.0, ymax, 0.0], [0.0, 0.0, 1.0]], float)
            self._inverted = None
            self._invalid = 0
        return self._mtx