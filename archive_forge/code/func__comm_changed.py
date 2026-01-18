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
@observe('comm')
def _comm_changed(self, change):
    """Called when the comm is changed."""
    if change['new'] is None:
        return
    self._model_id = self.model_id
    self.comm.on_msg(self._handle_msg)
    _instances[self.model_id] = self