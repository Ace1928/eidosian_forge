import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree
def _calc_vertices_regions(self):
    """
        Calculates the Voronoi vertices and regions of the generators stored
        in self.points. The vertices will be stored in self.vertices and the
        regions in self.regions.

        This algorithm was discussed at PyData London 2015 by
        Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk
        """
    conv = scipy.spatial.ConvexHull(self.points)
    self.vertices = self.radius * conv.equations[:, :-1] + self.center
    self._simplices = conv.simplices
    simplex_indices = np.arange(len(self._simplices))
    tri_indices = np.column_stack([simplex_indices] * self._dim).ravel()
    point_indices = self._simplices.ravel()
    indices = np.argsort(point_indices, kind='mergesort')
    flattened_groups = tri_indices[indices].astype(np.intp)
    intervals = np.cumsum(np.bincount(point_indices + 1))
    groups = [list(flattened_groups[intervals[i]:intervals[i + 1]]) for i in range(len(intervals) - 1)]
    self.regions = groups