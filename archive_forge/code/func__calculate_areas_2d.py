import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree
def _calculate_areas_2d(self):
    arcs = self.points[self._simplices] - self.center
    d = np.sum((arcs[:, 1] - arcs[:, 0]) ** 2, axis=1)
    theta = np.arccos(1 - d / (2 * self.radius ** 2))
    areas = self.radius * theta
    signs = np.sign(np.einsum('ij,ij->i', arcs[:, 0], self.vertices - self.center))
    indices = np.where(signs < 0)
    areas[indices] = 2 * np.pi * self.radius - areas[indices]
    return areas