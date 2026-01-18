from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase
def _resolve_ref(self, pname, value):
    if pname == 'object' and self.applies(value):
        return (None, value)
    return super()._resolve_ref(pname, value)