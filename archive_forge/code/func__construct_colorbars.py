from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
@param.depends('color_mappers')
def _construct_colorbars(self, color_mappers=None):
    if not color_mappers:
        color_mappers = self.color_mappers
    from bokeh.models import ColorBar, FixedTicker, Plot
    cbs = []
    for color_mapper in color_mappers:
        ticks = np.linspace(color_mapper.low, color_mapper.high, 5)
        cbs.append(ColorBar(color_mapper=color_mapper, title=color_mapper.name, ticker=FixedTicker(ticks=ticks), label_standoff=5, background_fill_alpha=0, orientation='horizontal', location=(0, 0)))
    plot = Plot(toolbar_location=None, frame_height=0, sizing_mode='stretch_width', outline_line_width=0)
    [plot.add_layout(cb, 'below') for cb in cbs]
    return plot