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
def _wrap_document_callback(self, callback: Callback) -> Callback:
    if getattr(callback, 'nolock', False):
        return callback

    def wrapped_callback(*args: Any, **kwargs: Any):
        return self.with_document_locked(callback, *args, **kwargs)
    return wrapped_callback