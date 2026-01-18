import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
class ScalarDSArtist(DSArtist):

    def __init__(self, ax, df, glyph, aggregator, agg_hook=None, shade_hook=None, plot_width=None, plot_height=None, x_range=None, y_range=None, width_scale=1.0, height_scale=1.0, norm=None, cmap=None, alpha=None, **kwargs):
        super().__init__(ax, df, glyph, aggregator, agg_hook, shade_hook, plot_width, plot_height, x_range, y_range, width_scale, height_scale, **kwargs)
        self._vmin = norm.vmin
        self._vmax = norm.vmax
        self.set_norm(norm)
        self.set_cmap(cmap)
        self.set_alpha(alpha)
        binned = self.aggregate(self.axes.get_xlim(), self.axes.get_ylim())
        if self.agg_hook is not None:
            binned = self.agg_hook(binned)
        self.set_ds_data(binned)
        self.set_array(np.eye(2))

    def shade(self, binned):
        mask = compute_mask(binned.data)
        A = np.ma.masked_array(binned.data, mask)
        self.set_array(A)
        if self._vmin is not None:
            self.norm.vmin = self._vmin
        if self._vmax is not None:
            self.norm.vmax = self._vmax
        if A.size:
            if self._vmin is None:
                self.norm.vmin = A.min()
            if self._vmax is None:
                self.norm.vmax = A.max()
        self.autoscale_None()
        return self.to_rgba(A, bytes=True, norm=True)

    def get_ds_image(self):
        binned = self.get_ds_data()
        rgba = self.to_rgba(self.get_array(), bytes=True, norm=True)
        return to_ds_image(binned, rgba)

    def get_legend_elements(self):
        return None