import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
@staticmethod
def _call_widget_constructed(widget):
    """Static method, called when a widget is constructed."""
    if Widget._widget_construction_callback is not None and callable(Widget._widget_construction_callback):
        Widget._widget_construction_callback(widget)