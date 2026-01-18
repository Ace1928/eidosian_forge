import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _get_colormapper(self, eldim, element, ranges, style, factors=None, colors=None, group=None, name='color_mapper'):
    if eldim is None and colors is None:
        return None
    dim_name = dim_range_key(eldim)
    if self.adjoined:
        cmappers = self.adjoined.traverse(lambda x: (x.handles.get('color_dim'), x.handles.get(name), [v for v in x.handles.values() if isinstance(v, ColorMapper)]))
        cmappers = [(cmap, mappers) for cdim, cmap, mappers in cmappers if cdim == eldim]
        if cmappers:
            cmapper, mappers = cmappers[0]
            if not cmapper:
                if mappers and mappers[0]:
                    cmapper = mappers[0]
                else:
                    return None
            self.handles['color_mapper'] = cmapper
            return cmapper
        else:
            return None
    ncolors = None if factors is None else len(factors)
    if eldim:
        if all((util.isfinite(cl) for cl in self.clim)):
            low, high = self.clim
        elif dim_name in ranges:
            if self.clim_percentile and 'robust' in ranges[dim_name]:
                low, high = ranges[dim_name]['robust']
            else:
                low, high = ranges[dim_name]['combined']
            dlow, dhigh = ranges[dim_name]['data']
            if util.is_int(low, int_like=True) and util.is_int(high, int_like=True) and util.is_int(dlow) and util.is_int(dhigh):
                low, high = (int(low), int(high))
        elif isinstance(eldim, dim):
            low, high = (np.nan, np.nan)
        else:
            low, high = element.range(eldim.name)
        if self.symmetric:
            sym_max = max(abs(low), high)
            low, high = (-sym_max, sym_max)
        low = self.clim[0] if util.isfinite(self.clim[0]) else low
        high = self.clim[1] if util.isfinite(self.clim[1]) else high
    else:
        low, high = (None, None)
    prefix = '' if group is None else group + '_'
    cmap = colors or style.get(prefix + 'cmap', style.get('cmap', 'viridis'))
    nan_colors = {k: rgba_tuple(v) for k, v in self.clipping_colors.items()}
    if isinstance(cmap, dict):
        factors = list(cmap)
        palette = [cmap.get(f, nan_colors.get('NaN', self._default_nan)) for f in factors]
        if isinstance(eldim, dim):
            if eldim.dimension in element:
                formatter = element.get_dimension(eldim.dimension).pprint_value
            else:
                formatter = str
        else:
            formatter = eldim.pprint_value
        factors = [formatter(f) for f in factors]
    else:
        categorical = ncolors is not None
        if isinstance(self.color_levels, int):
            ncolors = self.color_levels
        elif isinstance(self.color_levels, list):
            ncolors = len(self.color_levels) - 1
            if isinstance(cmap, list) and len(cmap) != ncolors:
                raise ValueError('The number of colors in the colormap must match the intervals defined in the color_levels, expected %d colors found %d.' % (ncolors, len(cmap)))
        palette = process_cmap(cmap, ncolors, categorical=categorical)
        if isinstance(self.color_levels, list):
            palette, (low, high) = color_intervals(palette, self.color_levels, clip=(low, high))
    colormapper, opts = self._get_cmapper_opts(low, high, factors, nan_colors)
    cmapper = self.handles.get(name)
    if cmapper is not None:
        if cmapper.palette != palette:
            cmapper.palette = palette
        opts = {k: opt for k, opt in opts.items() if getattr(cmapper, k) != opt}
        if opts:
            cmapper.update(**opts)
    else:
        cmapper = colormapper(palette=palette, **opts)
        self.handles[name] = cmapper
        self.handles['color_dim'] = eldim
    return cmapper