import numpy as np
import param
from ...core.options import SkipRendering
from ...core.util import isfinite
from ...element import Image, Raster
from ..mixins import HeatMapMixin
from .element import ColorbarPlot
class HeatMapPlot(HeatMapMixin, RasterPlot):

    def init_layout(self, key, element, ranges, **kwargs):
        layout = super().init_layout(key, element, ranges)
        gridded = element.gridded
        xdim, ydim = gridded.dimensions()[:2]
        if self.invert_axes:
            xaxis, yaxis = ('yaxis', 'xaxis')
        else:
            xaxis, yaxis = ('xaxis', 'yaxis')
        shape = gridded.interface.shape(gridded, gridded=True)
        xtype = gridded.interface.dtype(gridded, xdim)
        if xtype.kind in 'SUO':
            layout[xaxis]['tickvals'] = np.arange(shape[1])
            layout[xaxis]['ticktext'] = gridded.dimension_values(0, expanded=False)
        ytype = gridded.interface.dtype(gridded, ydim)
        if ytype.kind in 'SUO':
            layout[yaxis]['tickvals'] = np.arange(shape[0])
            layout[yaxis]['ticktext'] = gridded.dimension_values(1, expanded=False)
        return layout

    def get_data(self, element, ranges, style, **kwargs):
        if not element._unique:
            self.param.warning('HeatMap element index is not unique,  ensure you aggregate the data before displaying it, e.g. using heatmap.aggregate(function=np.mean). Duplicate index values have been dropped.')
        gridded = element.gridded
        xdim, ydim = gridded.dimensions()[:2]
        data = gridded.dimension_values(2, flat=False)
        xtype = gridded.interface.dtype(gridded, xdim)
        if xtype.kind in 'SUO':
            xvals = np.arange(data.shape[1] + 1) - 0.5
        else:
            xvals = gridded.interface.coords(gridded, xdim, edges=True, ordered=True)
        ytype = gridded.interface.dtype(gridded, ydim)
        if ytype.kind in 'SUO':
            yvals = np.arange(data.shape[0] + 1) - 0.5
        else:
            yvals = gridded.interface.coords(gridded, ydim, edges=True, ordered=True)
        if self.invert_axes:
            xvals, yvals = (yvals, xvals)
            data = data.T
        return [dict(x=xvals, y=yvals, z=data)]