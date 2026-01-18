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
class TransformedBbox(BboxBase):
    """
    A `Bbox` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """

    def __init__(self, bbox, transform, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
        transform : `Transform`
        """
        if not bbox.is_bbox:
            raise ValueError("'bbox' is not a bbox")
        _api.check_isinstance(Transform, transform=transform)
        if transform.input_dims != 2 or transform.output_dims != 2:
            raise ValueError("The input and output dimensions of 'transform' must be 2")
        super().__init__(**kwargs)
        self._bbox = bbox
        self._transform = transform
        self.set_children(bbox, transform)
        self._points = None
    __str__ = _make_str_method('_bbox', '_transform')

    def get_points(self):
        if self._invalid:
            p = self._bbox.get_points()
            points = self._transform.transform([[p[0, 0], p[0, 1]], [p[1, 0], p[0, 1]], [p[0, 0], p[1, 1]], [p[1, 0], p[1, 1]]])
            points = np.ma.filled(points, 0.0)
            xs = (min(points[:, 0]), max(points[:, 0]))
            if p[0, 0] > p[1, 0]:
                xs = xs[::-1]
            ys = (min(points[:, 1]), max(points[:, 1]))
            if p[0, 1] > p[1, 1]:
                ys = ys[::-1]
            self._points = np.array([[xs[0], ys[0]], [xs[1], ys[1]]])
            self._invalid = 0
        return self._points
    if DEBUG:
        _get_points = get_points

        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points

    def contains(self, x, y):
        return self._bbox.contains(*self._transform.inverted().transform((x, y)))

    def fully_contains(self, x, y):
        return self._bbox.fully_contains(*self._transform.inverted().transform((x, y)))