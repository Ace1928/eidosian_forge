import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class HeatMapMixin:

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        if range_type in ('data', 'combined'):
            agg = element.gridded
            xtype = agg.interface.dtype(agg, 0)
            shape = agg.interface.shape(agg, gridded=True)
            if xtype.kind in 'SUO':
                x0, x1 = (0 - 0.5, shape[1] - 0.5)
            else:
                x0, x1 = element.range(0)
            ytype = agg.interface.dtype(agg, 1)
            if ytype.kind in 'SUO':
                y0, y1 = (-0.5, shape[0] - 0.5)
            else:
                y0, y1 = element.range(1)
            return (x0, y0, x1, y1)
        else:
            return super().get_extents(element, ranges, range_type)