import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
class OffsetImage(OffsetBox):

    def __init__(self, arr, *, zoom=1, cmap=None, norm=None, interpolation=None, origin=None, filternorm=True, filterrad=4.0, resample=False, dpi_cor=True, **kwargs):
        super().__init__()
        self._dpi_cor = dpi_cor
        self.image = BboxImage(bbox=self.get_window_extent, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, filternorm=filternorm, filterrad=filterrad, resample=resample, **kwargs)
        self._children = [self.image]
        self.set_zoom(zoom)
        self.set_data(arr)

    def set_data(self, arr):
        self._data = np.asarray(arr)
        self.image.set_data(self._data)
        self.stale = True

    def get_data(self):
        return self._data

    def set_zoom(self, zoom):
        self._zoom = zoom
        self.stale = True

    def get_zoom(self):
        return self._zoom

    def get_offset(self):
        """Return offset of the container."""
        return self._offset

    def get_children(self):
        return [self.image]

    def get_bbox(self, renderer):
        dpi_cor = renderer.points_to_pixels(1.0) if self._dpi_cor else 1.0
        zoom = self.get_zoom()
        data = self.get_data()
        ny, nx = data.shape[:2]
        w, h = (dpi_cor * nx * zoom, dpi_cor * ny * zoom)
        return Bbox.from_bounds(0, 0, w, h)

    def draw(self, renderer):
        self.image.draw(renderer)
        self.stale = False