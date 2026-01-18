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
@staticmethod
def _rgb2hex(r, g, b):
    int_type = (int, np.integer)
    if isinstance(r, int_type) and isinstance(g, int_type) is isinstance(b, int_type):
        return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)
    else:
        return '#{0:02x}{1:02x}{2:02x}'.format(int(255 * r), int(255 * g), int(255 * b))