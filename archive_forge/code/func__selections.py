from __future__ import annotations
import re
import sys
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import lazy_load
from .base import ModelPane
@property
def _selections(self):
    return _get_selections(self.object)