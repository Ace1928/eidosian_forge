import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class PiecewiseAffineTransform(_GeometricTransform):
    """Piecewise affine transformation.

    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.

    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.

    """

    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, D) array_like
            Source coordinates.
        dst : (N, D) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if all pieces of the model are successfully estimated.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        ndim = src.shape[1]
        self._tesselation = spatial.Delaunay(src)
        success = True
        self.affines = []
        for tri in self._tesselation.simplices:
            affine = AffineTransform(dimensionality=ndim)
            success &= affine.estimate(src[tri, :], dst[tri, :])
            self.affines.append(affine)
        self._inverse_tesselation = spatial.Delaunay(dst)
        self.inverse_affines = []
        for tri in self._inverse_tesselation.simplices:
            affine = AffineTransform(dimensionality=ndim)
            success &= affine.estimate(dst[tri, :], src[tri, :])
            self.inverse_affines.append(affine)
        return success

    def __call__(self, coords):
        """Apply forward transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        coords = np.asarray(coords)
        out = np.empty_like(coords, np.float64)
        simplex = self._tesselation.find_simplex(coords)
        out[simplex == -1, :] = -1
        for index in range(len(self._tesselation.simplices)):
            affine = self.affines[index]
            index_mask = simplex == index
            out[index_mask, :] = affine(coords[index_mask, :])
        return out

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        tform = type(self)()
        tform._tesselation = self._inverse_tesselation
        tform._inverse_tesselation = self._tesselation
        tform.affines = self.inverse_affines
        tform.inverse_affines = self.affines
        return tform