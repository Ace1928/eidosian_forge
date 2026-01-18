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
def _get_schema_version(obj, default_version: int=5) -> int:
    if Vega.is_altair(obj):
        schema = obj.to_dict().get('$schema', '')
    else:
        schema = obj.get('$schema', '')
    version = schema.split('/')[-1]
    match = SCHEMA_REGEX.fullmatch(version)
    if match is None or not match.groups():
        return default_version
    return int(match.groups()[0])