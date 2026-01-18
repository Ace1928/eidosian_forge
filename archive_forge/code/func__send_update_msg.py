from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
def _send_update_msg(self, restyle_data, relayout_data, trace_indexes=None, source_view_id=None):
    if source_view_id:
        return
    trace_indexes = self._figure._normalize_trace_indexes(trace_indexes)
    msg = {}
    if relayout_data:
        msg['relayout'] = relayout_data
    if restyle_data:
        msg['restyle'] = {'data': restyle_data, 'traces': trace_indexes}
    for ref, (m, _) in self._models.items():
        self._apply_update([], msg, m, ref)