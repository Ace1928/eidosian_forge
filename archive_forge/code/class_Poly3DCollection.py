import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Poly3DCollection(PolyCollection):
    """
    A collection of 3D polygons.

    .. note::
        **Filling of 3D polygons**

        There is no simple definition of the enclosed surface of a 3D polygon
        unless the polygon is planar.

        In practice, Matplotlib fills the 2D projection of the polygon. This
        gives a correct filling appearance only for planar polygons. For all
        other polygons, you'll find orientations in which the edges of the
        polygon intersect in the projection. This will lead to an incorrect
        visualization of the 3D area.

        If you need filled areas, it is recommended to create them via
        `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
        triangulation and thus generates consistent surfaces.
    """

    def __init__(self, verts, *args, zsort='average', shade=False, lightsource=None, **kwargs):
        """
        Parameters
        ----------
        verts : list of (N, 3) array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (N, 3).
        zsort : {'average', 'min', 'max'}, default: 'average'
            The calculation method for the z-order.
            See `~.Poly3DCollection.set_zsort` for details.
        shade : bool, default: False
            Whether to shade *facecolors* and *edgecolors*. When activating
            *shade*, *facecolors* and/or *edgecolors* must be provided.

            .. versionadded:: 3.7

        lightsource : `~matplotlib.colors.LightSource`, optional
            The lightsource to use when *shade* is True.

            .. versionadded:: 3.7

        *args, **kwargs
            All other parameters are forwarded to `.PolyCollection`.

        Notes
        -----
        Note that this class does a bit of magic with the _facecolors
        and _edgecolors properties.
        """
        if shade:
            normals = _generate_normals(verts)
            facecolors = kwargs.get('facecolors', None)
            if facecolors is not None:
                kwargs['facecolors'] = _shade_colors(facecolors, normals, lightsource)
            edgecolors = kwargs.get('edgecolors', None)
            if edgecolors is not None:
                kwargs['edgecolors'] = _shade_colors(edgecolors, normals, lightsource)
            if facecolors is None and edgecolors is None:
                raise ValueError('You must provide facecolors, edgecolors, or both for shade to work.')
        super().__init__(verts, *args, **kwargs)
        if isinstance(verts, np.ndarray):
            if verts.ndim != 3:
                raise ValueError('verts must be a list of (N, 3) array-like')
        elif any((len(np.shape(vert)) != 2 for vert in verts)):
            raise ValueError('verts must be a list of (N, 3) array-like')
        self.set_zsort(zsort)
        self._codes3d = None
    _zsort_functions = {'average': np.average, 'min': np.min, 'max': np.max}

    def set_zsort(self, zsort):
        """
        Set the calculation method for the z-order.

        Parameters
        ----------
        zsort : {'average', 'min', 'max'}
            The function applied on the z-coordinates of the vertices in the
            viewer's coordinate system, to determine the z-order.
        """
        self._zsortfunc = self._zsort_functions[zsort]
        self._sort_zpos = None
        self.stale = True

    def get_vector(self, segments3d):
        """Optimize points for projection."""
        if len(segments3d):
            xs, ys, zs = np.vstack(segments3d).T
        else:
            xs, ys, zs = ([], [], [])
        ones = np.ones(len(xs))
        self._vec = np.array([xs, ys, zs, ones])
        indices = [0, *np.cumsum([len(segment) for segment in segments3d])]
        self._segslices = [*map(slice, indices[:-1], indices[1:])]

    def set_verts(self, verts, closed=True):
        """
        Set 3D vertices.

        Parameters
        ----------
        verts : list of (N, 3) array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (N, 3).
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        """
        self.get_vector(verts)
        super().set_verts([], False)
        self._closed = closed

    def set_verts_and_codes(self, verts, codes):
        """Set 3D vertices with path codes."""
        self.set_verts(verts, closed=False)
        self._codes3d = codes

    def set_3d_properties(self):
        self.update_scalarmappable()
        self._sort_zpos = None
        self.set_zsort('average')
        self._facecolor3d = PolyCollection.get_facecolor(self)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)
        self._alpha3d = PolyCollection.get_alpha(self)
        self.stale = True

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def do_3d_projection(self):
        """
        Perform the 3D projection for this object.
        """
        if self._A is not None:
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors
        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)
        if xyzlist:
            z_segments_2d = sorted(((self._zsortfunc(zs), np.column_stack([xs, ys]), fc, ec, idx) for idx, ((xs, ys, zs), fc, ec) in enumerate(zip(xyzlist, cface, cedge))), key=lambda x: x[0], reverse=True)
            _, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = zip(*z_segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []
        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)
        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            return np.min(tzs)
        else:
            return np.nan

    def set_facecolor(self, colors):
        super().set_facecolor(colors)
        self._facecolor3d = PolyCollection.get_facecolor(self)

    def set_edgecolor(self, colors):
        super().set_edgecolor(colors)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)

    def set_alpha(self, alpha):
        artist.Artist.set_alpha(self, alpha)
        try:
            self._facecolor3d = mcolors.to_rgba_array(self._facecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        try:
            self._edgecolors = mcolors.to_rgba_array(self._edgecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        self.stale = True

    def get_facecolor(self):
        if not hasattr(self, '_facecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._facecolors2d)

    def get_edgecolor(self):
        if not hasattr(self, '_edgecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._edgecolors2d)