import codecs
import os
import pydevd
import socket
import sys
import threading
import debugpy
from debugpy import adapter
from debugpy.common import json, log, sockets
from _pydevd_bundle.pydevd_constants import get_global_debugger
from pydevd_file_utils import absolute_path
from debugpy.common.util import hide_debugpy_internals
def _settrace(*args, **kwargs):
    log.debug('pydevd.settrace(*{0!r}, **{1!r})', args, kwargs)
    kwargs.setdefault('notify_stdin', False)
    try:
        return pydevd.settrace(*args, **kwargs)
    except Exception:
        raise
    else:
        _settrace.called = True