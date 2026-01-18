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
def _clean_relayout_data(relayout_data):
    return {key: val for key, val in relayout_data.items() if not key.endswith('._derived')}