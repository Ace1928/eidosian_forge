import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Line3DCollection(LineCollection):
    """
    A collection of 3D lines.
    """

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_segments(self, segments):
        """
        Set 3D segments.
        """
        self._segments3d = segments
        super().set_segments([])

    def do_3d_projection(self):
        """
        Project the points according to renderer matrix.
        """
        xyslist = [proj3d._proj_trans_points(points, self.axes.M) for points in self._segments3d]
        segments_2d = [np.column_stack([xs, ys]) for xs, ys, zs in xyslist]
        LineCollection.set_segments(self, segments_2d)
        minz = 1000000000.0
        for xs, ys, zs in xyslist:
            minz = min(minz, min(zs))
        return minz