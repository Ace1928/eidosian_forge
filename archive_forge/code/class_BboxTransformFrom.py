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
class BboxTransformFrom(Affine2DBase):
    """
    `BboxTransformFrom` linearly transforms points from a given `Bbox` to the
    unit bounding box.
    """
    is_separable = True

    def __init__(self, boxin, **kwargs):
        if not boxin.is_bbox:
            raise ValueError("'boxin' must be bbox")
        super().__init__(**kwargs)
        self._boxin = boxin
        self.set_children(boxin)
        self._mtx = None
        self._inverted = None
    __str__ = _make_str_method('_boxin')

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            if DEBUG and (inw == 0 or inh == 0):
                raise ValueError('Transforming from a singular bounding box.')
            x_scale = 1.0 / inw
            y_scale = 1.0 / inh
            self._mtx = np.array([[x_scale, 0.0, -inl * x_scale], [0.0, y_scale, -inb * y_scale], [0.0, 0.0, 1.0]], float)
            self._inverted = None
            self._invalid = 0
        return self._mtx