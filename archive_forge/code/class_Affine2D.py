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
class Affine2D(Affine2DBase):
    """
    A mutable 2D affine transformation.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Initialize an Affine transform from a 3x3 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        super().__init__(**kwargs)
        if matrix is None:
            matrix = IdentityTransform._mtx
        self._mtx = matrix.copy()
        self._invalid = 0
    _base_str = _make_str_method('_mtx')

    def __str__(self):
        return self._base_str() if (self._mtx != np.diag(np.diag(self._mtx))).any() else f'Affine2D().scale({self._mtx[0, 0]}, {self._mtx[1, 1]})' if self._mtx[0, 0] != self._mtx[1, 1] else f'Affine2D().scale({self._mtx[0, 0]})'

    @staticmethod
    def from_values(a, b, c, d, e, f):
        """
        Create a new Affine2D instance from the given values::

          a c e
          b d f
          0 0 1

        .
        """
        return Affine2D(np.array([a, c, e, b, d, f, 0.0, 0.0, 1.0], float).reshape((3, 3)))

    def get_matrix(self):
        """
        Get the underlying transformation matrix as a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        if self._invalid:
            self._inverted = None
            self._invalid = 0
        return self._mtx

    def set_matrix(self, mtx):
        """
        Set the underlying transformation matrix from a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        self._mtx = mtx
        self.invalidate()

    def set(self, other):
        """
        Set this transformation from the frozen copy of another
        `Affine2DBase` object.
        """
        _api.check_isinstance(Affine2DBase, other=other)
        self._mtx = other.get_matrix()
        self.invalidate()

    def clear(self):
        """
        Reset the underlying matrix to the identity transform.
        """
        self._mtx = IdentityTransform._mtx.copy()
        self.invalidate()
        return self

    def rotate(self, theta):
        """
        Add a rotation (in radians) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        a = math.cos(theta)
        b = math.sin(theta)
        mtx = self._mtx
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        mtx[0, 0] = a * xx - b * yx
        mtx[0, 1] = a * xy - b * yy
        mtx[0, 2] = a * x0 - b * y0
        mtx[1, 0] = b * xx + a * yx
        mtx[1, 1] = b * xy + a * yy
        mtx[1, 2] = b * x0 + a * y0
        self.invalidate()
        return self

    def rotate_deg(self, degrees):
        """
        Add a rotation (in degrees) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.rotate(math.radians(degrees))

    def rotate_around(self, x, y, theta):
        """
        Add a rotation (in radians) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """
        Add a rotation (in degrees) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        x, y = (float(x), float(y))
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)

    def translate(self, tx, ty):
        """
        Add a translation in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        self._mtx[0, 2] += tx
        self._mtx[1, 2] += ty
        self.invalidate()
        return self

    def scale(self, sx, sy=None):
        """
        Add a scale in place.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        if sy is None:
            sy = sx
        self._mtx[0, 0] *= sx
        self._mtx[0, 1] *= sx
        self._mtx[0, 2] *= sx
        self._mtx[1, 0] *= sy
        self._mtx[1, 1] *= sy
        self._mtx[1, 2] *= sy
        self.invalidate()
        return self

    def skew(self, xShear, yShear):
        """
        Add a skew in place.

        *xShear* and *yShear* are the shear angles along the *x*- and
        *y*-axes, respectively, in radians.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        rx = math.tan(xShear)
        ry = math.tan(yShear)
        mtx = self._mtx
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        mtx[0, 0] += rx * yx
        mtx[0, 1] += rx * yy
        mtx[0, 2] += rx * y0
        mtx[1, 0] += ry * xx
        mtx[1, 1] += ry * xy
        mtx[1, 2] += ry * x0
        self.invalidate()
        return self

    def skew_deg(self, xShear, yShear):
        """
        Add a skew in place.

        *xShear* and *yShear* are the shear angles along the *x*- and
        *y*-axes, respectively, in degrees.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.skew(math.radians(xShear), math.radians(yShear))