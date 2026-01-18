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
def _create_param_pane(inst, widgets_kwargs=None, parameters=None):
    widgets_kwargs = widgets_kwargs or {}
    for pname in inst.param:
        if pname == 'name':
            continue
        if pname not in widgets_kwargs:
            widgets_kwargs[pname] = {}
        if isinstance(inst.param[pname], (param.Number, param.Range)):
            widgets_kwargs[pname]['throttled'] = True
        widgets_kwargs[pname]['sizing_mode'] = 'stretch_width'
    kwargs = {'show_name': False, 'widgets': widgets_kwargs, 'width': CONTROLS_WIDTH}
    if parameters:
        kwargs['parameters'] = parameters
    pane = pn.Param(inst.param, **kwargs)
    return pane