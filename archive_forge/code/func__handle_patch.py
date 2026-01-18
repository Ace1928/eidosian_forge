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
@_needs_document_lock
def _handle_patch(self, message: msg.patch_doc, connection: ServerConnection) -> msg.ok:
    self._current_patch_connection = connection
    try:
        message.apply_to_document(self.document, self)
    finally:
        self._current_patch_connection = None
    return connection.ok(message)