from __future__ import annotations
import errno
import importlib.util
import os
import socket
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, NewType, Sequence
from urllib.parse import (
from urllib.parse import (
from urllib.request import pathname2url as _pathname2url
from _frozen_importlib_external import _NamespacePath
from jupyter_core.utils import ensure_async as _ensure_async
from packaging.version import Version
from tornado.httpclient import AsyncHTTPClient, HTTPClient, HTTPRequest, HTTPResponse
from tornado.netutil import Resolver
def _check_pid_posix(pid: int) -> bool:
    """Copy of IPython.utils.process.check_pid"""
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            return True
        raise
    else:
        return True