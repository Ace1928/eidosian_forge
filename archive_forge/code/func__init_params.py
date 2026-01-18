from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
def _init_params(self):
    viewport_params = [p for p in self.param if 'viewport' in p]
    parameters = list(Layoutable.param) + viewport_params
    params = {p: getattr(self, p) for p in parameters if getattr(self, p) is not None}
    if self.object is None:
        json, sources = ({}, [])
    else:
        fig = self._to_figure(self.object)
        json = self._plotly_json_wrapper(fig)
        sources = Plotly._get_sources(json)
    params['_render_count'] = self._render_count
    params['config'] = self.config or {}
    params['data'] = json.get('data', [])
    params['data_sources'] = sources
    params['layout'] = layout = json.get('layout', {})
    params['frames'] = json.get('frames', [])
    if layout.get('autosize') and self.sizing_mode is self.param.sizing_mode.default:
        params['sizing_mode'] = 'stretch_both'
        if 'styles' not in params:
            params['styles'] = {}
    return params