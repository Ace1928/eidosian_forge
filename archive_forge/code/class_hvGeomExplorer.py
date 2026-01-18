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
class hvGeomExplorer(hvPlotExplorer):
    kind = param.Selector(default=None, objects=KINDS['all'])

    @property
    def _var_name(self):
        return 'gdf'

    @property
    def _single_y(self):
        return True

    @property
    def _x(self):
        return None

    @property
    def _y(self):
        return None

    @param.depends('x')
    def xlim(self):
        pass

    @param.depends('y')
    def ylim(self):
        pass

    @property
    def _groups(self):
        return ['gridded', 'dataframe']