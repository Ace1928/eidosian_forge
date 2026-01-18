import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
class FancyArrow(Polygon):
    """
    Like Arrow, but lets you set head width and head height independently.
    """
    _edge_default = True

    def __str__(self):
        return 'FancyArrow()'

    @_docstring.dedent_interpd
    def __init__(self, x, y, dx, dy, *, width=0.001, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, **kwargs):
        """
        Parameters
        ----------
        x, y : float
            The x and y coordinates of the arrow base.

        dx, dy : float
            The length of the arrow along x and y direction.

        width : float, default: 0.001
            Width of full arrow tail.

        length_includes_head : bool, default: False
            True if head is to be counted in calculating the length.

        head_width : float or None, default: 3*width
            Total width of the full arrow head.

        head_length : float or None, default: 1.5*head_width
            Length of arrow head.

        shape : {'full', 'left', 'right'}, default: 'full'
            Draw the left-half, right-half, or full arrow.

        overhang : float, default: 0
            Fraction that the arrow is swept back (0 overhang means
            triangular shape). Can be negative or greater than one.

        head_starts_at_zero : bool, default: False
            If True, the head starts being drawn at coordinate 0
            instead of ending at coordinate 0.

        **kwargs
            `.Patch` properties:

            %(Patch:kwdoc)s
        """
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._width = width
        self._length_includes_head = length_includes_head
        self._head_width = head_width
        self._head_length = head_length
        self._shape = shape
        self._overhang = overhang
        self._head_starts_at_zero = head_starts_at_zero
        self._make_verts()
        super().__init__(self.verts, closed=True, **kwargs)

    def set_data(self, *, x=None, y=None, dx=None, dy=None, width=None, head_width=None, head_length=None):
        """
        Set `.FancyArrow` x, y, dx, dy, width, head_with, and head_length.
        Values left as None will not be updated.

        Parameters
        ----------
        x, y : float or None, default: None
            The x and y coordinates of the arrow base.

        dx, dy : float or None, default: None
            The length of the arrow along x and y direction.

        width : float or None, default: None
            Width of full arrow tail.

        head_width : float or None, default: None
            Total width of the full arrow head.

        head_length : float or None, default: None
            Length of arrow head.
        """
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        if dx is not None:
            self._dx = dx
        if dy is not None:
            self._dy = dy
        if width is not None:
            self._width = width
        if head_width is not None:
            self._head_width = head_width
        if head_length is not None:
            self._head_length = head_length
        self._make_verts()
        self.set_xy(self.verts)

    def _make_verts(self):
        if self._head_width is None:
            head_width = 3 * self._width
        else:
            head_width = self._head_width
        if self._head_length is None:
            head_length = 1.5 * head_width
        else:
            head_length = self._head_length
        distance = np.hypot(self._dx, self._dy)
        if self._length_includes_head:
            length = distance
        else:
            length = distance + head_length
        if not length:
            self.verts = np.empty([0, 2])
        else:
            hw, hl = (head_width, head_length)
            hs, lw = (self._overhang, self._width)
            left_half_arrow = np.array([[0.0, 0.0], [-hl, -hw / 2], [-hl * (1 - hs), -lw / 2], [-length, -lw / 2], [-length, 0]])
            if not self._length_includes_head:
                left_half_arrow += [head_length, 0]
            if self._head_starts_at_zero:
                left_half_arrow += [head_length / 2, 0]
            if self._shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow * [1, -1]
                if self._shape == 'right':
                    coords = right_half_arrow
                elif self._shape == 'full':
                    coords = np.concatenate([left_half_arrow[:-1], right_half_arrow[-2::-1]])
                else:
                    raise ValueError(f'Got unknown shape: {self._shape!r}')
            if distance != 0:
                cx = self._dx / distance
                sx = self._dy / distance
            else:
                cx, sx = (0, 1)
            M = [[cx, sx], [-sx, cx]]
            self.verts = np.dot(coords, M) + [self._x + self._dx, self._y + self._dy]