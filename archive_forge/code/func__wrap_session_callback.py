from __future__ import annotations
import logging # isort:skip
import inspect
import time
from copy import copy
from functools import wraps
from typing import (
from tornado import locks
from ..events import ConnectionLost
from ..util.token import generate_jwt_token
from .callbacks import DocumentCallbackGroup
def _wrap_session_callback(self, callback: SessionCallback) -> SessionCallback:
    wrapped = copy(callback)
    wrapped._callback = self._wrap_document_callback(callback.callback)
    return wrapped