import numpy as np
import param
from param.parameterized import bothmethod
from ..core import Dataset, Operation
from ..core.util import datetime_types, dt_to_int, isfinite, max_range
from ..element import Image
from ..streams import PlotSize, RangeX, RangeXY
def _dt_transform(self, x_range, y_range, xs, ys, xtype, ytype):
    (xstart, xend), (ystart, yend) = (x_range, y_range)
    if xtype == 'datetime':
        xstart, xend = np.array([xstart, xend]).astype('datetime64[ns]')
        xs = xs.astype('datetime64[ns]')
    if ytype == 'datetime':
        ystart, yend = np.array([ystart, yend]).astype('datetime64[ns]')
        ys = ys.astype('datetime64[ns]')
    return (((xstart, xend), (ystart, yend)), (xs, ys))