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
class CategoricalDSArtist(DSArtist):

    def __init__(self, ax, df, glyph, aggregator, agg_hook=None, shade_hook=None, plot_width=None, plot_height=None, x_range=None, y_range=None, width_scale=1.0, height_scale=1.0, color_key=None, alpha_range=(40, 255), color_baseline=None, **kwargs):
        super().__init__(ax, df, glyph, aggregator, agg_hook, shade_hook, plot_width, plot_height, x_range, y_range, width_scale, height_scale, **kwargs)
        self._color_key = color_key
        self._alpha_range = alpha_range
        self._color_baseline = color_baseline
        binned = self.aggregate(self.axes.get_xlim(), self.axes.get_ylim())
        if self.agg_hook is not None:
            binned = self.agg_hook(binned)
        self.set_ds_data(binned)
        self.set_array(np.eye(2))

    def shade(self, binned):
        img = tf.shade(binned, color_key=self._color_key, min_alpha=self._alpha_range[0], alpha=self._alpha_range[1], color_baseline=self._color_baseline)
        rgba = uint32_to_uint8(img.data)
        return rgba

    def get_ds_image(self):
        binned = self.get_ds_data()
        rgba = self.get_array()
        return to_ds_image(binned, rgba)

    def get_legend_elements(self):
        """
        Return legend elements to display the color code for each category.
        """
        if not isinstance(self._color_key, dict):
            binned = self.get_ds_data()
            categories = binned.coords[binned.dims[2]].data
            color_dict = dict(zip(categories, self._color_key))
        else:
            color_dict = self._color_key
        return [Patch(facecolor=color, edgecolor='none', label=category) for category, color in color_dict.items()]