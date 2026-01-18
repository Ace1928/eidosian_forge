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
def _vtklut2bkcmap(self, lut, name):
    table = lut.GetTable()
    low, high = lut.GetTableRange()
    rgba_arr = np.frombuffer(memoryview(table), dtype=np.uint8).reshape((-1, 4))
    palette = [self._rgb2hex(*rgb) for rgb in rgba_arr[:, :3]]
    return LinearColorMapper(low=low, high=high, name=name, palette=palette)