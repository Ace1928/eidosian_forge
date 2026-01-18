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
class BlendedAffine2D(_BlendedMixin, Affine2DBase):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This version is an optimization for the case where both child
    transforms are of type `Affine2DBase`.
    """
    is_separable = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        Both *x_transform* and *y_transform* must be 2D affine transforms.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        is_affine = x_transform.is_affine and y_transform.is_affine
        is_separable = x_transform.is_separable and y_transform.is_separable
        is_correct = is_affine and is_separable
        if not is_correct:
            raise ValueError('Both *x_transform* and *y_transform* must be 2D affine transforms')
        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)
        Affine2DBase.__init__(self)
        self._mtx = None

    def get_matrix(self):
        if self._invalid:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()
            else:
                x_mtx = self._x.get_matrix()
                y_mtx = self._y.get_matrix()
                self._mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
            self._inverted = None
            self._invalid = 0
        return self._mtx