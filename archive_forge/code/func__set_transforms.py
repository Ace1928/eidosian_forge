import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_transforms(self):
    """Calculate transforms immediately before drawing."""
    ax = self.axes
    fig = self.figure
    if self._units == 'xy':
        sc = 1
    elif self._units == 'x':
        sc = ax.bbox.width / ax.viewLim.width
    elif self._units == 'y':
        sc = ax.bbox.height / ax.viewLim.height
    elif self._units == 'inches':
        sc = fig.dpi
    elif self._units == 'points':
        sc = fig.dpi / 72.0
    elif self._units == 'width':
        sc = ax.bbox.width
    elif self._units == 'height':
        sc = ax.bbox.height
    elif self._units == 'dots':
        sc = 1.0
    else:
        raise ValueError(f'Unrecognized units: {self._units!r}')
    self._transforms = np.zeros((len(self._widths), 3, 3))
    widths = self._widths * sc
    heights = self._heights * sc
    sin_angle = np.sin(self._angles)
    cos_angle = np.cos(self._angles)
    self._transforms[:, 0, 0] = widths * cos_angle
    self._transforms[:, 0, 1] = heights * -sin_angle
    self._transforms[:, 1, 0] = widths * sin_angle
    self._transforms[:, 1, 1] = heights * cos_angle
    self._transforms[:, 2, 2] = 1.0
    _affine = transforms.Affine2D
    if self._units == 'xy':
        m = ax.transData.get_affine().get_matrix().copy()
        m[:2, 2:] = 0
        self.set_transform(_affine(m))