from collections import defaultdict
import numpy as np
import param
from ...core import util
from ...element import Contours, Polygons
from ...util.transform import dim
from .callbacks import PolyDrawCallback, PolyEditCallback
from .element import ColorbarPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import multi_polygons_data
class ContourPlot(PathPlot):
    selected = param.List(default=None, doc='\n        The current selection as a list of integers corresponding\n        to the selected items.')
    show_legend = param.Boolean(default=False, doc='\n        Whether to show legend for the plot.')
    color_index = param.ClassSelector(default=0, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `color=dim('color')`")
    _color_style = 'line_color'
    _nonvectorized_styles = base_properties + ['cmap']

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self._has_holes = None

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims) + self.hmap.last.last.vdims
        else:
            dims = list(self.overlay_dims.keys()) + self.hmap.last.vdims
        return (dims, {})

    def _get_hover_data(self, data, element):
        """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
        if 'hover' not in self.handles or self.static_source:
            return
        interface = element.interface
        scalar_kwargs = {'per_geom': True} if interface.multi else {}
        for d in element.vdims:
            dim = util.dimension_sanitizer(d.name)
            if dim not in data:
                if interface.isunique(element, d, **scalar_kwargs):
                    data[dim] = element.dimension_values(d, expanded=False)
                else:
                    data[dim] = element.split(datatype='array', dimensions=[d])
        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v] * len(next(iter(data.values())))

    def get_data(self, element, ranges, style):
        if self._has_holes is None:
            draw_callbacks = any((isinstance(cb, (PolyDrawCallback, PolyEditCallback)) for cb in self.callbacks))
            has_holes = isinstance(element, Polygons) and (not draw_callbacks)
            self._has_holes = has_holes
        else:
            has_holes = self._has_holes
        if not element.interface.multi:
            element = element.clone([element.data], datatype=type(element).datatype)
        if self.static_source:
            data = {}
            xs = self.handles['cds'].data['xs']
        else:
            if has_holes:
                xs, ys = multi_polygons_data(element)
            else:
                xs, ys = (list(element.dimension_values(kd, expanded=False)) for kd in element.kdims)
            if self.invert_axes:
                xs, ys = (ys, xs)
            data = dict(xs=xs, ys=ys)
        mapping = dict(self._mapping)
        self._get_hover_data(data, element)
        color, fill_color = (style.get('color'), style.get('fill_color'))
        if (isinstance(color, dim) and color.applies(element) or color in element) or (isinstance(fill_color, dim) and fill_color.applies(element)) or fill_color in element:
            cdim = None
        else:
            cidx = self.color_index + 2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)
        if cdim is None:
            return (data, mapping, style)
        dim_name = util.dimension_sanitizer(cdim.name)
        values = element.dimension_values(cdim, expanded=False)
        data[dim_name] = values
        factors = None
        if cdim.name in ranges and 'factors' in ranges[cdim.name]:
            factors = ranges[cdim.name]['factors']
        elif values.dtype.kind in 'SUO' and len(values):
            if isinstance(values[0], np.ndarray):
                values = np.concatenate(values)
            factors = util.unique_array(values)
        cmapper = self._get_colormapper(cdim, element, ranges, style, factors)
        mapping[self._color_style] = {'field': dim_name, 'transform': cmapper}
        if self.show_legend:
            mapping['legend_field'] = dim_name
        return (data, mapping, style)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        plot_method = properties.pop('plot_method', None)
        properties = mpl_to_bokeh(properties)
        data = dict(properties, **mapping)
        if self._has_holes:
            plot_method = 'multi_polygons'
        elif plot_method is None:
            plot_method = self._plot_methods.get('single')
        renderer = getattr(plot, plot_method)(**data)
        if self.colorbar:
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k[:-12])
        return (renderer, renderer.glyph)