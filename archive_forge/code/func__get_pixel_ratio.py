import numpy as np
import param
from param.parameterized import bothmethod
from ..core import Dataset, Operation
from ..core.util import datetime_types, dt_to_int, isfinite, max_range
from ..element import Image
from ..streams import PlotSize, RangeX, RangeXY
def _get_pixel_ratio(self):
    if self.p.pixel_ratio is None:
        from panel import state
        if state.browser_info and isinstance(state.browser_info.device_pixel_ratio, (int, float)):
            return state.browser_info.device_pixel_ratio
        else:
            return 1
    else:
        return self.p.pixel_ratio