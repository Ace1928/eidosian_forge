import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
class AxLine(Line2D):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self, xy1, xy2, slope, **kwargs):
        """
        Parameters
        ----------
        xy1 : (float, float)
            The first set of (x, y) coordinates for the line to pass through.
        xy2 : (float, float) or None
            The second set of (x, y) coordinates for the line to pass through.
            Both *xy2* and *slope* must be passed, but one of them must be None.
        slope : float or None
            The slope of the line. Both *xy2* and *slope* must be passed, but one of
            them must be None.
        """
        super().__init__([0, 1], [0, 1], **kwargs)
        if xy2 is None and slope is None or (xy2 is not None and slope is not None):
            raise TypeError("Exactly one of 'xy2' and 'slope' must be given")
        self._slope = slope
        self._xy1 = xy1
        self._xy2 = xy2

    def get_transform(self):
        ax = self.axes
        points_transform = self._transform - ax.transData + ax.transScale
        if self._xy2 is not None:
            (x1, y1), (x2, y2) = points_transform.transform([self._xy1, self._xy2])
            dx = x2 - x1
            dy = y2 - y1
            if np.allclose(x1, x2):
                if np.allclose(y1, y2):
                    raise ValueError(f'Cannot draw a line through two identical points (x={(x1, x2)}, y={(y1, y2)})')
                slope = np.inf
            else:
                slope = dy / dx
        else:
            x1, y1 = points_transform.transform(self._xy1)
            slope = self._slope
        (vxlo, vylo), (vxhi, vyhi) = ax.transScale.transform(ax.viewLim)
        if np.isclose(slope, 0):
            start = (vxlo, y1)
            stop = (vxhi, y1)
        elif np.isinf(slope):
            start = (x1, vylo)
            stop = (x1, vyhi)
        else:
            _, start, stop, _ = sorted([(vxlo, y1 + (vxlo - x1) * slope), (vxhi, y1 + (vxhi - x1) * slope), (x1 + (vylo - y1) / slope, vylo), (x1 + (vyhi - y1) / slope, vyhi)])
        return BboxTransformTo(Bbox([start, stop])) + ax.transLimits + ax.transAxes

    def draw(self, renderer):
        self._transformed_path = None
        super().draw(renderer)

    def get_xy1(self):
        """
        Return the *xy1* value of the line.
        """
        return self._xy1

    def get_xy2(self):
        """
        Return the *xy2* value of the line.
        """
        return self._xy2

    def get_slope(self):
        """
        Return the *slope* value of the line.
        """
        return self._slope

    def set_xy1(self, x, y):
        """
        Set the *xy1* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
        self._xy1 = (x, y)

    def set_xy2(self, x, y):
        """
        Set the *xy2* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
        if self._slope is None:
            self._xy2 = (x, y)
        else:
            raise ValueError("Cannot set an 'xy2' value while 'slope' is set; they differ but their functionalities overlap")

    def set_slope(self, slope):
        """
        Set the *slope* value of the line.

        Parameters
        ----------
        slope : float
            The slope of the line.
        """
        if self._xy2 is None:
            self._slope = slope
        else:
            raise ValueError("Cannot set a 'slope' value while 'xy2' is set; they differ but their functionalities overlap")