from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
@param.depends('object', 'link_figure', watch=True)
def _update_figure(self):
    import plotly.graph_objs as go
    if self.object is None or type(self.object) is not go.Figure or self.object is self._figure or (not self.link_figure):
        return
    fig = self.object
    fig._send_addTraces_msg = lambda *_, **__: self._update_from_figure('add')
    fig._send_moveTraces_msg = lambda *_, **__: self._update_from_figure('move')
    fig._send_deleteTraces_msg = lambda *_, **__: self._update_from_figure('delete')
    fig._send_restyle_msg = self._send_restyle_msg
    fig._send_relayout_msg = self._send_relayout_msg
    fig._send_update_msg = self._send_update_msg
    fig._send_animate_msg = lambda *_, **__: self._update_from_figure('animate')
    self._figure = fig