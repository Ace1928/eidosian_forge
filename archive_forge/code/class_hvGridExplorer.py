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
class hvGridExplorer(hvPlotExplorer):
    kind = param.Selector(default='image', objects=KINDS['all'])

    def __init__(self, ds, **params):
        import xarray as xr
        var_name_suffix = ''
        if isinstance(ds, xr.Dataset):
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1:
                ds = ds[data_vars[0]]
                var_name_suffix = f"['{data_vars[0]}']"
            else:
                ds = ds.to_array('variable').transpose(..., 'variable')
                var_name_suffix = ".to_array('variable').transpose(..., 'variable')"
        if 'kind' not in params:
            params['kind'] = 'image'
        self._var_name_suffix = var_name_suffix
        super().__init__(ds, **params)

    @property
    def _var_name(self):
        if self._var_name_suffix:
            return f'ds{self._var_name_suffix}'
        else:
            return 'da'

    @property
    def _x(self):
        return self._converter.x or self._converter.indexes[0] if self.x is None else self.x

    @property
    def _y(self):
        return self._converter.y or self._converter.indexes[1] if self.y is None else self.y

    @param.depends('x')
    def xlim(self):
        try:
            values = self._data[self._x]
        except:
            return (0, 1)
        if values.dtype.kind in 'OSU':
            return None
        return (np.nanmin(values), np.nanmax(values))

    @param.depends('y', 'y_multi')
    def ylim(self):
        y = self._y
        if not isinstance(y, list):
            y = [y]
        values = (self._data[y] for y in y)
        return max_range([(np.nanmin(vs), np.nanmax(vs)) for vs in values])

    @property
    def _groups(self):
        return ['gridded', 'dataframe', 'geom']

    def _populate(self):
        variables = self._converter.variables
        indexes = getattr(self._converter, 'indexes', [])
        variables_no_index = [v for v in variables if v not in indexes]
        for pname in self.param:
            if pname == 'kind':
                continue
            p = self.param[pname]
            if isinstance(p, param.Selector):
                if pname in ['x', 'y', 'groupby', 'by']:
                    p.objects = indexes
                else:
                    p.objects = variables_no_index
                if pname == 'x' and getattr(self, pname, None) is None:
                    setattr(self, pname, p.objects[0])
                elif pname == 'y' and getattr(self, pname, None) is None:
                    setattr(self, pname, p.objects[1])
                elif pname == 'groupby' and len(getattr(self, pname, [])) == 0 and (len(p.objects) > 2):
                    setattr(self, pname, p.objects[2:])