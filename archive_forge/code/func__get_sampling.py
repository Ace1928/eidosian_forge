import numpy as np
import param
from param.parameterized import bothmethod
from ..core import Dataset, Operation
from ..core.util import datetime_types, dt_to_int, isfinite, max_range
from ..element import Image
from ..streams import PlotSize, RangeX, RangeXY
def _get_sampling(self, element, x, y, ndim=2, default=None):
    target = self.p.target
    if not isinstance(x, list) and x is not None:
        x = [x]
    if not isinstance(y, list) and y is not None:
        y = [y]
    if target:
        x0, y0, x1, y1 = target.bounds.lbrt()
        x_range, y_range = ((x0, x1), (y0, y1))
        height, width = target.dimension_values(2, flat=False).shape
    else:
        if x is None:
            x_range = self.p.x_range or (-0.5, 0.5)
        elif self.p.expand or not self.p.x_range:
            if self.p.x_range and all((isfinite(v) for v in self.p.x_range)):
                x_range = self.p.x_range
            else:
                x_range = max_range([element.range(xd) for xd in x])
        else:
            x0, x1 = self.p.x_range
            ex0, ex1 = max_range([element.range(xd) for xd in x])
            x_range = (np.nanmin([np.nanmax([x0, ex0]), ex1]), np.nanmax([np.nanmin([x1, ex1]), ex0]))
        if y is None and ndim == 2:
            y_range = self.p.y_range or default or (-0.5, 0.5)
        elif self.p.expand or not self.p.y_range:
            if self.p.y_range and all((isfinite(v) for v in self.p.y_range)):
                y_range = self.p.y_range
            elif default is None:
                y_range = max_range([element.range(yd) for yd in y])
            else:
                y_range = default
        else:
            y0, y1 = self.p.y_range
            if default is None:
                ey0, ey1 = max_range([element.range(yd) for yd in y])
            else:
                ey0, ey1 = default
            y_range = (np.nanmin([np.nanmax([y0, ey0]), ey1]), np.nanmax([np.nanmin([y1, ey1]), ey0]))
        width, height = (self.p.width, self.p.height)
    (xstart, xend), (ystart, yend) = (x_range, y_range)
    xtype = 'numeric'
    if isinstance(xstart, str) or isinstance(xend, str):
        raise ValueError('Categorical data is not supported')
    elif isinstance(xstart, datetime_types) or isinstance(xend, datetime_types):
        xstart, xend = (dt_to_int(xstart, 'ns'), dt_to_int(xend, 'ns'))
        xtype = 'datetime'
    elif not np.isfinite(xstart) and (not np.isfinite(xend)):
        xstart, xend = (0, 0)
        if x and element.get_dimension_type(x[0]) in datetime_types:
            xtype = 'datetime'
    ytype = 'numeric'
    if isinstance(ystart, str) or isinstance(yend, str):
        raise ValueError('Categorical data is not supported')
    elif isinstance(ystart, datetime_types) or isinstance(yend, datetime_types):
        ystart, yend = (dt_to_int(ystart, 'ns'), dt_to_int(yend, 'ns'))
        ytype = 'datetime'
    elif not np.isfinite(ystart) and (not np.isfinite(yend)):
        ystart, yend = (0, 0)
        if y and element.get_dimension_type(y[0]) in datetime_types:
            ytype = 'datetime'
    xspan = xend - xstart
    yspan = yend - ystart
    if self.p.x_sampling:
        width = int(min([xspan / self.p.x_sampling, width]))
    if self.p.y_sampling:
        height = int(min([yspan / self.p.y_sampling, height]))
    if xstart == xend or width == 0:
        xunit, width = (0, 0)
    else:
        xunit = float(xspan) / width
    if ystart == yend or height == 0:
        yunit, height = (0, 0)
    else:
        yunit = float(yspan) / height
    xs, ys = (np.linspace(xstart + xunit / 2.0, xend - xunit / 2.0, width), np.linspace(ystart + yunit / 2.0, yend - yunit / 2.0, height))
    pixel_ratio = self._get_pixel_ratio()
    width = int(width * pixel_ratio)
    height = int(height * pixel_ratio)
    return (((xstart, xend), (ystart, yend)), (xs, ys), (width, height), (xtype, ytype))