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
def _get_selections(obj, version=None):
    if obj is None:
        return {}
    elif version is None:
        version = _get_schema_version(obj)
    key = 'params' if version >= 5 else 'selection'
    selections = {}
    if _isin(obj, key):
        params = obj[key]
        if version >= 5 and isinstance(params, list):
            params = {p.name if hasattr(p, 'name') else p['name']: p for p in params if getattr(p, 'param_type', None) == 'selection' or _isin(p, 'select')}
        try:
            selections.update({name: _get_type(spec, version) for name, spec in params.items()})
        except (AttributeError, TypeError):
            pass
    for c in _containers:
        if _isin(obj, c):
            for subobj in obj[c]:
                selections.update(_get_selections(subobj, version=version))
    return selections