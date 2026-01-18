import sys
import numpy as np
import param
from bokeh.models import CustomJSHover, DatetimeAxis
from ...core.util import cartesian_product, dimension_sanitizer, isfinite
from ...element import Raster
from ..util import categorical_legend
from .chart import PointPlot
from .element import ColorbarPlot, LegendPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, mpl_to_bokeh
from .util import bokeh33, bokeh34, colormesh
def _collect_hover_data(self, element, mask=(), irregular=False):
    """
        Returns a dict mapping hover dimension names to flattened arrays.

        Note that `Quad` glyphs are used when given 1-D coords but `Patches` are
        used for "irregular" 2-D coords, and Bokeh inserts data into these glyphs
        in the opposite order such that the relationship b/w the `invert_axes`
        parameter and the need to transpose the arrays before flattening is
        reversed.
        """
    transpose = self.invert_axes if irregular else not self.invert_axes
    hover_dims = element.dimensions()[3:]
    hover_vals = [element.dimension_values(hover_dim, flat=False) for hover_dim in hover_dims]
    hover_data = {}
    for hdim, hvals in zip(hover_dims, hover_vals):
        hdat = hvals.T.flatten() if transpose else hvals.flatten()
        hover_data[dimension_sanitizer(hdim.name)] = hdat[mask]
    return hover_data