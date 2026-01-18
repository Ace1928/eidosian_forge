from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
def _send_restyle_msg(self, restyle_data, trace_indexes=None, source_view_id=None):
    self._send_update_msg(restyle_data, {}, trace_indexes, source_view_id)