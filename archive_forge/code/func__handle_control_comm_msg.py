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
@classmethod
def _handle_control_comm_msg(cls, msg):
    if cls._control_comm is None:
        raise RuntimeError('Control comm has not been properly opened')
    data = msg['content']['data']
    method = data['method']
    if method == 'request_states':
        cls.get_manager_state()
        widgets = _instances.values()
        full_state = {}
        drop_defaults = False
        for widget in widgets:
            full_state[widget.model_id] = {'model_name': widget._model_name, 'model_module': widget._model_module, 'model_module_version': widget._model_module_version, 'state': widget.get_state(drop_defaults=drop_defaults)}
        full_state, buffer_paths, buffers = _remove_buffers(full_state)
        cls._control_comm.send(dict(method='update_states', states=full_state, buffer_paths=buffer_paths), buffers=buffers)
    else:
        raise RuntimeError('Unknown front-end to back-end widget control msg with method "%s"' % method)