from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase
def _ipywidget_transform(obj):
    """
    Transforms an ipywidget into a Parameter that listens updates
    when the ipywidget updates.
    """
    if not (IPyWidget.applies(obj) and hasattr(obj, 'value')):
        return obj
    name = type(obj).__name__
    if name in _ipywidget_classes:
        ipy_param = _ipywidget_classes[name]
    else:
        ipy_param = param.parameterized_class(name, {'value': param.Parameter()})
    _ipywidget_classes[name] = ipy_param
    ipy_inst = ipy_param(value=obj.value)
    obj.observe(lambda event: ipy_inst.param.update(value=event['new']), 'value')
    return ipy_inst.param.value