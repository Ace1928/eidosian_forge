import holoviews as _hv
import numpy as np
import panel as pn
import param
from holoviews.core.util import datetime_types, dt_to_int, is_number, max_range
from holoviews.element import tile_sources
from holoviews.plotting.util import list_cmaps
from panel.viewable import Viewer
from .converter import HoloViewsConverter as _hvConverter
from .plotting import hvPlot as _hvPlot
from .util import is_geodataframe, is_xarray, instantiate_crs_str
@param.depends('color', 'explorer.kind', 'symmetric', watch=True)
def _update_coloropts(self):
    if not self.colormapped or self.cmap not in list(DEFAULT_CMAPS.values()):
        return
    if self.explorer.kind in _hvConverter._colorbar_types:
        key = 'diverging' if self.symmetric else 'linear'
        self.colorbar = True
    elif self.color in self._data:
        kind = self._data[self.color].dtype.kind
        if kind in 'OSU':
            key = 'categorical'
        elif self.symmetric:
            key = 'diverging'
        else:
            key = 'linear'
    else:
        return
    self.cmap = DEFAULT_CMAPS[key]