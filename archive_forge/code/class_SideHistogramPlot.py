from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
class SideHistogramPlot(HistogramPlot):
    style_opts = HistogramPlot.style_opts + ['cmap']
    height = param.Integer(default=125, doc='The height of the plot')
    width = param.Integer(default=125, doc='The width of the plot')
    show_title = param.Boolean(default=False, doc='\n        Whether to display the plot title.')
    default_tools = param.List(default=['save', 'pan', 'wheel_zoom', 'box_zoom', 'reset'], doc='A list of plugin tools to use on the plot.')
    _callback = "\n    color_mapper.low = cb_obj['geometry']['{axis}0'];\n    color_mapper.high = cb_obj['geometry']['{axis}1'];\n    source.change.emit()\n    main_source.change.emit()\n    "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.invert_axes:
            self.default_tools.append('ybox_select')
        else:
            self.default_tools.append('xbox_select')

    def get_data(self, element, ranges, style):
        data, mapping, style = HistogramPlot.get_data(self, element, ranges, style)
        color_dims = [d for d in self.adjoined.traverse(lambda x: x.handles.get('color_dim')) if d is not None]
        dimension = color_dims[0] if color_dims else None
        cmapper = self._get_colormapper(dimension, element, {}, {})
        if cmapper:
            cvals = None
            if isinstance(dimension, dim):
                if dimension.applies(element):
                    dim_name = dimension.dimension.name
                    cvals = [] if self.static_source else dimension.apply(element)
            elif dimension in element.dimensions():
                dim_name = dimension.name
                cvals = [] if self.static_source else element.dimension_values(dimension)
            if cvals is not None:
                data[dim_name] = cvals
                mapping['fill_color'] = {'field': dim_name, 'transform': cmapper}
        return (data, mapping, style)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        ret = super()._init_glyph(plot, mapping, properties)
        if 'field' not in mapping.get('fill_color', {}):
            return ret
        dim = mapping['fill_color']['field']
        sources = self.adjoined.traverse(lambda x: (x.handles.get('color_dim'), x.handles.get('source')))
        sources = [src for cdim, src in sources if cdim == dim]
        tools = [t for t in self.handles['plot'].tools if isinstance(t, BoxSelectTool)]
        if not tools or not sources:
            return
        main_source = sources[0]
        handles = {'color_mapper': self.handles['color_mapper'], 'source': self.handles['source'], 'cds': self.handles['source'], 'main_source': main_source}
        callback = self._callback.format(axis='y' if self.invert_axes else 'x')
        self.state.js_on_event('selectiongeometry', CustomJS(args=handles, code=callback))
        return ret