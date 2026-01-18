from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def _indexes_changed(self, old, new):
    """
        Comparator that checks whether DataFrame indexes have changed.

        If indexes and length are unchanged we assume we do not
        have to reset various settings including expanded rows,
        scroll position, pagination etc.
        """
    if type(old) != type(new) or isinstance(new, dict):
        return True
    elif len(old) != len(new):
        return False
    return (old.index != new.index).any()