from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase
def _get_ipywidget(self, obj, doc: Document, root: Model, comm: Optional[Comm], **kwargs):
    if not isinstance(comm, JupyterComm) or 'PANEL_IPYWIDGET' in os.environ:
        from ..io.ipywidget import Widget
    import reacton
    widget, rc = reacton.render(obj)
    self._rcs[root.ref['id']] = rc
    return super()._get_ipywidget(widget, doc, root, comm, **kwargs)