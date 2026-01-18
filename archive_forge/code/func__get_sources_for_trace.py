from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
@staticmethod
def _get_sources_for_trace(json, data, parent_path=''):
    for key, value in list(json.items()):
        full_path = key if not parent_path else '{}.{}'.format(parent_path, key)
        if isinstance(value, np.ndarray):
            data[full_path] = [json.pop(key)]
        elif isinstance(value, dict):
            Plotly._get_sources_for_trace(value, data=data, parent_path=full_path)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, element in enumerate(value):
                element_path = full_path + '.' + str(i)
                Plotly._get_sources_for_trace(element, data=data, parent_path=element_path)