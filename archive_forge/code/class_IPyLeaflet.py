from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase
class IPyLeaflet(IPyWidget):
    sizing_mode = param.ObjectSelector(default='stretch_width', objects=['fixed', 'stretch_width', 'stretch_height', 'stretch_both', 'scale_width', 'scale_height', 'scale_both', None])
    priority: float | bool | None = 0.7

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return IPyWidget.applies(obj) and obj._view_module == 'jupyter-leaflet'