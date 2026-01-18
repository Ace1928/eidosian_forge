from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from . import constants  # noqa
from .constants import *  # noqa
from zmq.backend import *  # noqa
from zmq import sugar
from zmq.sugar import *  # noqa
@contextmanager
def _libs_on_path():
    """context manager for libs directory on $PATH

    Works around mysterious issue where os.add_dll_directory
    does not resolve imports (conda-forge Python >= 3.8)
    """
    if not sys.platform.startswith('win'):
        yield
        return
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pyzmq.libs'))
    if not os.path.exists(libs_dir):
        yield
        return
    path_before = os.environ.get('PATH')
    try:
        os.environ['PATH'] = os.pathsep.join([path_before or '', libs_dir])
        yield
    finally:
        if path_before is None:
            os.environ.pop('PATH')
        else:
            os.environ['PATH'] = path_before