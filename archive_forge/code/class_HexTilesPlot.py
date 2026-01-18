from collections.abc import Callable
import numpy as np
import param
from bokeh.util.hex import cartesian_to_axial
from ...core import Dimension, Operation
from ...core.options import Compositor
from ...core.util import isfinite, max_range
from ...element import HexTiles
from ...util.transform import dim as dim_transform
from .element import ColorbarPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties
class HexTilesPlot(ColorbarPlot):
    aggregator = param.ClassSelector(default=np.size, class_=(Callable, tuple), doc='\n      Aggregation function or dimension transform used to compute\n      bin values.  Defaults to np.size to count the number of values\n      in each bin.')
    gridsize = param.ClassSelector(default=50, class_=(int, tuple), doc='\n      Number of hexagonal bins along x- and y-axes. Defaults to uniform\n      sampling along both axes when setting and integer but independent\n      bin sampling can be specified a tuple of integers corresponding to\n      the number of bins along each axis.')
    min_count = param.Number(default=None, doc='\n      The display threshold before a bin is shown, by default bins with\n      a count of less than 1 are hidden.')
    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'], doc='\n      The orientation of hexagon bins. By default the pointy side is on top.')
    color_index = param.ClassSelector(default=2, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `color=dim('color')`")
    max_scale = param.Number(default=0.9, bounds=(0, None), doc='\n      When size_index is enabled this defines the maximum size of each\n      bin relative to uniform tile size, i.e. for a value of 1, the\n      largest bin will match the size of bins when scaling is disabled.\n      Setting value larger than 1 will result in overlapping bins.')
    min_scale = param.Number(default=0, bounds=(0, None), doc='\n      When size_index is enabled this defines the minimum size of each\n      bin relative to uniform tile size, i.e. for a value of 1, the\n      smallest bin will match the size of bins when scaling is disabled.\n      Setting value larger than 1 will result in overlapping bins.')
    size_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc='\n      Index of the dimension from which the sizes will the drawn.')
    selection_display = BokehOverlaySelectionDisplay()
    style_opts = base_properties + line_properties + fill_properties + ['cmap', 'scale']
    _nonvectorized_styles = base_properties + ['cmap', 'line_dash']
    _plot_methods = dict(single='hex_tile')

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        xdim, ydim = element.kdims[:2]
        ranges[xdim.name]['data'] = xdim.range
        ranges[ydim.name]['data'] = ydim.range
        xd = element.cdims.get(xdim.name)
        if xd and xdim.name in ranges:
            ranges[xdim.name]['hard'] = xd.range
            ranges[xdim.name]['soft'] = max_range([xd.soft_range, ranges[xdim.name]['soft']])
        yd = element.cdims.get(ydim.name)
        if yd and ydim.name in ranges:
            ranges[ydim.name]['hard'] = yd.range
            ranges[ydim.name]['hard'] = max_range([yd.soft_range, ranges[ydim.name]['soft']])
        return super().get_extents(element, ranges, range_type)

    def _hover_opts(self, element):
        if self.aggregator is np.size:
            dims = [Dimension('Count')]
        else:
            dims = element.vdims
        return (dims, {})

    def get_data(self, element, ranges, style):
        mapping = {'q': 'q', 'r': 'r'}
        if not len(element):
            data = {'q': [], 'r': []}
            return (data, mapping, style)
        q, r = (element.dimension_values(i) for i in range(2))
        x, y = element.kdims[::-1] if self.invert_axes else element.kdims
        (x0, x1), (y0, y1) = (x.range, y.range)
        if isinstance(self.gridsize, tuple):
            sx, sy = self.gridsize
        else:
            sx, sy = (self.gridsize, self.gridsize)
        xsize = (x1 - x0) / sx * (2.0 / 3.0)
        ysize = (y1 - y0) / sy * (2.0 / 3.0)
        size = xsize if self.orientation == 'flat' else ysize
        scale = ysize / xsize
        data = {'q': q, 'r': r}
        cdata, cmapping = self._get_color_data(element, ranges, style)
        data.update(cdata)
        mapping.update(cmapping)
        if self.min_count is not None and self.min_count <= 0:
            cmapper = cmapping['color']['transform']
            cmapper.low = self.min_count
            self.state.background_fill_color = cmapper.palette[0]
        self._get_hover_data(data, element, element.vdims)
        style['orientation'] = self.orientation + 'top'
        style['size'] = size
        style['aspect_scale'] = scale
        scale_dim = element.get_dimension(self.size_index)
        scale = style.get('scale')
        if scale_dim and (isinstance(scale, str) and scale in element or isinstance(scale, dim_transform)):
            self.param.warning("Cannot declare style mapping for 'scale' option and declare a size_index; ignoring the size_index.")
            scale_dim = None
        if scale_dim is not None:
            sizes = element.dimension_values(scale_dim)
            if self.aggregator is np.size:
                ptp = sizes.max()
                baseline = 0
            else:
                ptp = sizes.ptp()
                baseline = sizes.min()
            if self.min_scale > self.max_scale:
                raise ValueError('min_scale parameter must be smaller than max_scale parameter.')
            scale = self.max_scale - self.min_scale
            mapping['scale'] = 'scale'
            data['scale'] = (sizes - baseline) / ptp * scale + self.min_scale
        return (data, mapping, style)