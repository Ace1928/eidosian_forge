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
class Colormapping(Controls):
    clim = param.Range(label='Colorbar Limits (clim)', doc='\n        Upper and lower limits of the colorbar.')
    cnorm = param.Selector(label='Colorbar Normalization (cnorm)', default='linear', objects=['linear', 'log', 'eq_hist'])
    color = param.String(default=None)
    colorbar = param.Boolean(default=None)
    cmap = param.Selector(default=DEFAULT_CMAPS['linear'], label='Colormap', objects=CMAPS)
    rescale_discrete_levels = param.Boolean(default=True)
    symmetric = param.Boolean(default=False, doc='\n        Whether the data are symmetric around zero.')
    _widgets_kwargs = {'clim': {'placeholder': '(min, max)'}}

    def __init__(self, data, **params):
        super().__init__(data, **params)
        if not 'symmetric' in params:
            self.symmetric = self.explorer._converter._plot_opts.get('symmetric', False)

    @property
    def colormapped(self):
        if self.explorer.kind in _hvConverter._colorbar_types:
            return True
        return self.color is not None and self.color in self._data

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