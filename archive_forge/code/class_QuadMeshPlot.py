import numpy as np
import param
from ...core.options import SkipRendering
from ...core.util import isfinite
from ...element import Image, Raster
from ..mixins import HeatMapMixin
from .element import ColorbarPlot
class QuadMeshPlot(RasterPlot):
    nodata = param.Integer(default=None, doc='\n        Optional missing-data value for integer data.\n        If non-None, data with this value will be replaced with NaN so\n        that it is transparent (by default) when plotted.')

    def get_data(self, element, ranges, style, **kwargs):
        x, y, z = element.dimensions()[:3]
        irregular = element.interface.irregular(element, x)
        if irregular:
            raise SkipRendering('Plotly QuadMeshPlot only supports rectilinear meshes')
        xc, yc = (element.interface.coords(element, x, edges=True, ordered=True), element.interface.coords(element, y, edges=True, ordered=True))
        zdata = element.dimension_values(z, flat=False)
        x, y = ('x', 'y')
        if self.invert_axes:
            y, x = ('x', 'y')
            zdata = zdata.T
        return [{x: xc, y: yc, 'z': zdata}]