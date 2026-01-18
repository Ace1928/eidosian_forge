import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
class PolyQuadMesh(_MeshData, PolyCollection):
    """
    Class for drawing a quadrilateral mesh as individual Polygons.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    Notes
    -----
    Unlike `.QuadMesh`, this class will draw each cell as an individual Polygon.
    This is significantly slower, but allows for more flexibility when wanting
    to add additional properties to the cells, such as hatching.

    Another difference from `.QuadMesh` is that if any of the vertices or data
    of a cell are masked, that Polygon will **not** be drawn and it won't be in
    the list of paths returned.
    """

    def __init__(self, coordinates, **kwargs):
        self._deprecated_compression = False
        super().__init__(coordinates=coordinates)
        PolyCollection.__init__(self, verts=[], **kwargs)
        self._original_mask = ~self._get_unmasked_polys()
        self._deprecated_compression = np.any(self._original_mask)
        self._set_unmasked_verts()

    def _get_unmasked_polys(self):
        """Get the unmasked regions using the coordinates and array"""
        mask = np.any(np.ma.getmaskarray(self._coordinates), axis=-1)
        mask = mask[0:-1, 0:-1] | mask[1:, 1:] | mask[0:-1, 1:] | mask[1:, 0:-1]
        if getattr(self, '_deprecated_compression', False) and np.any(self._original_mask):
            return ~(mask | self._original_mask)
        with cbook._setattr_cm(self, _deprecated_compression=False):
            arr = self.get_array()
        if arr is not None:
            arr = np.ma.getmaskarray(arr)
            if arr.ndim == 3:
                mask |= np.any(arr, axis=-1)
            elif arr.ndim == 2:
                mask |= arr
            else:
                mask |= arr.reshape(self._coordinates[:-1, :-1, :].shape[:2])
        return ~mask

    def _set_unmasked_verts(self):
        X = self._coordinates[..., 0]
        Y = self._coordinates[..., 1]
        unmask = self._get_unmasked_polys()
        X1 = np.ma.filled(X[:-1, :-1])[unmask]
        Y1 = np.ma.filled(Y[:-1, :-1])[unmask]
        X2 = np.ma.filled(X[1:, :-1])[unmask]
        Y2 = np.ma.filled(Y[1:, :-1])[unmask]
        X3 = np.ma.filled(X[1:, 1:])[unmask]
        Y3 = np.ma.filled(Y[1:, 1:])[unmask]
        X4 = np.ma.filled(X[:-1, 1:])[unmask]
        Y4 = np.ma.filled(Y[:-1, 1:])[unmask]
        npoly = len(X1)
        xy = np.ma.stack([X1, Y1, X2, Y2, X3, Y3, X4, Y4, X1, Y1], axis=-1)
        verts = xy.reshape((npoly, 5, 2))
        self.set_verts(verts)

    def get_edgecolor(self):
        ec = super().get_edgecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        if len(ec) != len(unmasked_polys):
            return ec
        return ec[unmasked_polys, :]

    def get_facecolor(self):
        fc = super().get_facecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        if len(fc) != len(unmasked_polys):
            return fc
        return fc[unmasked_polys, :]

    def set_array(self, A):
        prev_unmask = self._get_unmasked_polys()
        if self._deprecated_compression and np.ndim(A) == 1:
            _api.warn_deprecated('3.8', message=f'Setting a PolyQuadMesh array using the compressed values is deprecated. Pass the full 2D shape of the original array {prev_unmask.shape} including the masked elements.')
            Afull = np.empty(self._original_mask.shape)
            Afull[~self._original_mask] = A
            mask = self._original_mask.copy()
            mask[~self._original_mask] |= np.ma.getmask(A)
            A = np.ma.array(Afull, mask=mask)
            return super().set_array(A)
        self._deprecated_compression = False
        super().set_array(A)
        if not np.array_equal(prev_unmask, self._get_unmasked_polys()):
            self._set_unmasked_verts()

    def get_array(self):
        A = super().get_array()
        if A is None:
            return
        if self._deprecated_compression and np.any(np.ma.getmask(A)):
            _api.warn_deprecated('3.8', message='Getting the array from a PolyQuadMesh will return the full array in the future (uncompressed). To get this behavior now set the PolyQuadMesh with a 2D array .set_array(data2d).')
            return np.ma.compressed(A)
        return A