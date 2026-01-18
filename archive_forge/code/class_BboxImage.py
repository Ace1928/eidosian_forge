import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
class BboxImage(_ImageBase):
    """The Image class whose size is determined by the given bbox."""

    def __init__(self, bbox, *, cmap=None, norm=None, interpolation=None, origin=None, filternorm=True, filterrad=4.0, resample=False, **kwargs):
        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
        super().__init__(None, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, filternorm=filternorm, filterrad=filterrad, resample=resample, **kwargs)
        self.bbox = bbox

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure()._get_renderer()
        if isinstance(self.bbox, BboxBase):
            return self.bbox
        elif callable(self.bbox):
            return self.bbox(renderer)
        else:
            raise ValueError('Unknown type of bbox')

    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
        if self._different_canvas(mouseevent) or not self.get_visible():
            return (False, {})
        x, y = (mouseevent.x, mouseevent.y)
        inside = self.get_window_extent().contains(x, y)
        return (inside, {})

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        width, height = renderer.get_canvas_width_height()
        bbox_in = self.get_window_extent(renderer).frozen()
        bbox_in._points /= [width, height]
        bbox_out = self.get_window_extent(renderer)
        clip = Bbox([[0, 0], [width, height]])
        self._transform = BboxTransformTo(clip)
        return self._make_image(self._A, bbox_in, bbox_out, clip, magnification, unsampled=unsampled)