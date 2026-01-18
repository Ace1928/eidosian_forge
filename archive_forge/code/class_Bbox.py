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
class Bbox(BboxBase):
    """
    A mutable bounding box.

    Examples
    --------
    **Create from known bounds**

    The default constructor takes the boundary "points" ``[[xmin, ymin],
    [xmax, ymax]]``.

        >>> Bbox([[1, 1], [3, 7]])
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    Alternatively, a Bbox can be created from the flattened points array, the
    so-called "extents" ``(xmin, ymin, xmax, ymax)``

        >>> Bbox.from_extents(1, 1, 3, 7)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    or from the "bounds" ``(xmin, ymin, width, height)``.

        >>> Bbox.from_bounds(1, 1, 2, 6)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    **Create from collections of points**

    The "empty" object for accumulating Bboxs is the null bbox, which is a
    stand-in for the empty set.

        >>> Bbox.null()
        Bbox([[inf, inf], [-inf, -inf]])

    Adding points to the null bbox will give you the bbox of those points.

        >>> box = Bbox.null()
        >>> box.update_from_data_xy([[1, 1]])
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])
        >>> box.update_from_data_xy([[2, 3], [3, 2]], ignore=False)
        >>> box
        Bbox([[1.0, 1.0], [3.0, 3.0]])

    Setting ``ignore=True`` is equivalent to starting over from a null bbox.

        >>> box.update_from_data_xy([[1, 1]], ignore=True)
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])

    .. warning::

        It is recommended to always specify ``ignore`` explicitly.  If not, the
        default value of ``ignore`` can be changed at any time by code with
        access to your Bbox, for example using the method `~.Bbox.ignore`.

    **Properties of the ``null`` bbox**

    .. note::

        The current behavior of `Bbox.null()` may be surprising as it does
        not have all of the properties of the "empty set", and as such does
        not behave like a "zero" object in the mathematical sense. We may
        change that in the future (with a deprecation period).

    The null bbox is the identity for intersections

        >>> Bbox.intersection(Bbox([[1, 1], [3, 7]]), Bbox.null())
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    except with itself, where it returns the full space.

        >>> Bbox.intersection(Bbox.null(), Bbox.null())
        Bbox([[-inf, -inf], [inf, inf]])

    A union containing null will always return the full space (not the other
    set!)

        >>> Bbox.union([Bbox([[0, 0], [0, 0]]), Bbox.null()])
        Bbox([[-inf, -inf], [inf, inf]])
    """

    def __init__(self, points, **kwargs):
        """
        Parameters
        ----------
        points : `~numpy.ndarray`
            A (2, 2) array of the form ``[[x0, y0], [x1, y1]]``.
        """
        super().__init__(**kwargs)
        points = np.asarray(points, float)
        if points.shape != (2, 2):
            raise ValueError('Bbox points must be of the form "[[x0, y0], [x1, y1]]".')
        self._points = points
        self._minpos = _default_minpos.copy()
        self._ignore = True
        self._points_orig = self._points.copy()
    if DEBUG:
        ___init__ = __init__

        def __init__(self, points, **kwargs):
            self._check(points)
            self.___init__(points, **kwargs)

        def invalidate(self):
            self._check(self._points)
            super().invalidate()

    def frozen(self):
        frozen_bbox = super().frozen()
        frozen_bbox._minpos = self.minpos.copy()
        return frozen_bbox

    @staticmethod
    def unit():
        """Create a new unit `Bbox` from (0, 0) to (1, 1)."""
        return Bbox([[0, 0], [1, 1]])

    @staticmethod
    def null():
        """Create a new null `Bbox` from (inf, inf) to (-inf, -inf)."""
        return Bbox([[np.inf, np.inf], [-np.inf, -np.inf]])

    @staticmethod
    def from_bounds(x0, y0, width, height):
        """
        Create a new `Bbox` from *x0*, *y0*, *width* and *height*.

        *width* and *height* may be negative.
        """
        return Bbox.from_extents(x0, y0, x0 + width, y0 + height)

    @staticmethod
    def from_extents(*args, minpos=None):
        """
        Create a new Bbox from *left*, *bottom*, *right* and *top*.

        The *y*-axis increases upwards.

        Parameters
        ----------
        left, bottom, right, top : float
            The four extents of the bounding box.
        minpos : float or None
            If this is supplied, the Bbox will have a minimum positive value
            set. This is useful when dealing with logarithmic scales and other
            scales where negative bounds result in floating point errors.
        """
        bbox = Bbox(np.reshape(args, (2, 2)))
        if minpos is not None:
            bbox._minpos[:] = minpos
        return bbox

    def __format__(self, fmt):
        return 'Bbox(x0={0.x0:{1}}, y0={0.y0:{1}}, x1={0.x1:{1}}, y1={0.y1:{1}})'.format(self, fmt)

    def __str__(self):
        return format(self, '')

    def __repr__(self):
        return 'Bbox([[{0.x0}, {0.y0}], [{0.x1}, {0.y1}]])'.format(self)

    def ignore(self, value):
        """
        Set whether the existing bounds of the box should be ignored
        by subsequent calls to :meth:`update_from_data_xy`.

        value : bool
            - When ``True``, subsequent calls to `update_from_data_xy` will
              ignore the existing bounds of the `Bbox`.
            - When ``False``, subsequent calls to `update_from_data_xy` will
              include the existing bounds of the `Bbox`.
        """
        self._ignore = value

    def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
        """
        Update the bounds of the `Bbox` to contain the vertices of the
        provided path. After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        path : `~matplotlib.path.Path`
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
            When ``True``, update the x/y values.
        """
        if ignore is None:
            ignore = self._ignore
        if path.vertices.size == 0:
            return
        points, minpos, changed = update_path_extents(path, None, self._points, self._minpos, ignore)
        if changed:
            self.invalidate()
            if updatex:
                self._points[:, 0] = points[:, 0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:, 1] = points[:, 1]
                self._minpos[1] = minpos[1]

    def update_from_data_x(self, x, ignore=None):
        """
        Update the x-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *width*, and *x0* will be the
        minimal value.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Array of x-values.
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        """
        x = np.ravel(x)
        self.update_from_data_xy(np.column_stack([x, np.ones(x.size)]), ignore=ignore, updatey=False)

    def update_from_data_y(self, y, ignore=None):
        """
        Update the y-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *height*, and *y0* will be the
        minimal value.

        Parameters
        ----------
        y : `~numpy.ndarray`
            Array of y-values.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        """
        y = np.ravel(y)
        self.update_from_data_xy(np.column_stack([np.ones(y.size), y]), ignore=ignore, updatex=False)

    def update_from_data_xy(self, xy, ignore=None, updatex=True, updatey=True):
        """
        Update the `Bbox` bounds based on the passed in *xy* coordinates.

        After updating, the bounds will have positive *width* and *height*;
        *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        xy : (N, 2) array-like
            The (x, y) coordinates.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
             When ``True``, update the x/y values.
        """
        if len(xy) == 0:
            return
        path = Path(xy)
        self.update_from_path(path, ignore=ignore, updatex=updatex, updatey=updatey)

    @BboxBase.x0.setter
    def x0(self, val):
        self._points[0, 0] = val
        self.invalidate()

    @BboxBase.y0.setter
    def y0(self, val):
        self._points[0, 1] = val
        self.invalidate()

    @BboxBase.x1.setter
    def x1(self, val):
        self._points[1, 0] = val
        self.invalidate()

    @BboxBase.y1.setter
    def y1(self, val):
        self._points[1, 1] = val
        self.invalidate()

    @BboxBase.p0.setter
    def p0(self, val):
        self._points[0] = val
        self.invalidate()

    @BboxBase.p1.setter
    def p1(self, val):
        self._points[1] = val
        self.invalidate()

    @BboxBase.intervalx.setter
    def intervalx(self, interval):
        self._points[:, 0] = interval
        self.invalidate()

    @BboxBase.intervaly.setter
    def intervaly(self, interval):
        self._points[:, 1] = interval
        self.invalidate()

    @BboxBase.bounds.setter
    def bounds(self, bounds):
        l, b, w, h = bounds
        points = np.array([[l, b], [l + w, b + h]], float)
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    @property
    def minpos(self):
        """
        The minimum positive value in both directions within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum extent instead of *p0*.
        """
        return self._minpos

    @property
    def minposx(self):
        """
        The minimum positive value in the *x*-direction within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum *x*-extent instead of *x0*.
        """
        return self._minpos[0]

    @property
    def minposy(self):
        """
        The minimum positive value in the *y*-direction within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum *y*-extent instead of *y0*.
        """
        return self._minpos[1]

    def get_points(self):
        """
        Get the points of the bounding box as an array of the form
        ``[[x0, y0], [x1, y1]]``.
        """
        self._invalid = 0
        return self._points

    def set_points(self, points):
        """
        Set the points of the bounding box directly from an array of the form
        ``[[x0, y0], [x1, y1]]``.  No error checking is performed, as this
        method is mainly for internal use.
        """
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    def set(self, other):
        """
        Set this bounding box from the "frozen" bounds of another `Bbox`.
        """
        if np.any(self._points != other.get_points()):
            self._points = other.get_points()
            self.invalidate()

    def mutated(self):
        """Return whether the bbox has changed since init."""
        return self.mutatedx() or self.mutatedy()

    def mutatedx(self):
        """Return whether the x-limits have changed since init."""
        return self._points[0, 0] != self._points_orig[0, 0] or self._points[1, 0] != self._points_orig[1, 0]

    def mutatedy(self):
        """Return whether the y-limits have changed since init."""
        return self._points[0, 1] != self._points_orig[0, 1] or self._points[1, 1] != self._points_orig[1, 1]