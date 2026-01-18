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
@_show_traceback
def _handle_msg(self, msg):
    """Called when a msg is received from the front-end"""
    data = msg['content']['data']
    method = data['method']
    if method == 'update':
        if 'state' in data:
            state = data['state']
            if 'buffer_paths' in data:
                _put_buffers(state, data['buffer_paths'], msg['buffers'])
            self.set_state(state)
    elif method == 'request_state':
        self.send_state()
    elif method == 'custom':
        if 'content' in data:
            self._handle_custom_msg(data['content'], msg['buffers'])
    else:
        self.log.error('Unknown front-end to back-end widget msg with method "%s"' % method)