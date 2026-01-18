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
class hex_binning(Operation):
    """
    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    usable.
    """
    aggregator = param.ClassSelector(default=np.size, class_=(Callable, tuple), doc='\n      Aggregation function or dimension transform used to compute bin\n      values. Defaults to np.size to count the number of values\n      in each bin.')
    gridsize = param.ClassSelector(default=50, class_=(int, tuple))
    invert_axes = param.Boolean(default=False)
    min_count = param.Number(default=None)
    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'])

    def _process(self, element, key=None):
        gridsize, aggregator, orientation = (self.p.gridsize, self.p.aggregator, self.p.orientation)
        indexes = [1, 0] if self.p.invert_axes else [0, 1]
        (x0, x1), (y0, y1) = (element.range(i) for i in indexes)
        if isinstance(gridsize, tuple):
            sx, sy = gridsize
        else:
            sx, sy = (gridsize, gridsize)
        xsize = (x1 - x0) / sx * (2.0 / 3.0)
        ysize = (y1 - y0) / sy * (2.0 / 3.0)
        size = xsize if self.orientation == 'flat' else ysize
        if isfinite(ysize) and isfinite(xsize) and (not xsize == 0):
            scale = ysize / xsize
        else:
            scale = 1
        x, y = (element.dimension_values(i) for i in indexes)
        if not len(x):
            return element.clone([])
        finite = isfinite(x) & isfinite(y)
        x, y = (x[finite], y[finite])
        q, r = cartesian_to_axial(x, y, size, orientation + 'top', scale)
        coords = (q, r)
        if aggregator is np.size:
            aggregator = np.sum
            values = (np.full_like(q, 1),)
            vdims = ['Count']
        elif not element.vdims:
            raise ValueError('HexTiles aggregated by value must define a value dimensions.')
        else:
            vdims = element.vdims
            values = tuple((element.dimension_values(vdim) for vdim in vdims))
        data = coords + values
        xd, yd = (element.get_dimension(i) for i in indexes)
        xdn, ydn = (xd.clone(range=(x0, x1)), yd.clone(range=(y0, y1)))
        kdims = [ydn, xdn] if self.p.invert_axes else [xdn, ydn]
        agg = element.clone(data, kdims=kdims, vdims=vdims).aggregate(function=aggregator)
        if self.p.min_count is not None and self.p.min_count > 1:
            agg = agg[:, :, self.p.min_count:]
        agg.cdims = {xd.name: xdn, yd.name: ydn}
        return agg