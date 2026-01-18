from __future__ import annotations
import gc
import sys
import threading
import types
from contextlib import contextmanager
from enum import Enum
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Callable, Final
from blinker import Signal
from streamlit import config, runtime, source_util, util
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.logger import get_logger
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_requests import (
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.vendor.ipython.modified_sys_path import modified_sys_path
@contextmanager
def _set_execing_flag(self):
    """A context for setting the ScriptRunner._execing flag.

        Used by _maybe_handle_execution_control_request to ensure that
        we only handle requests while we're inside an exec() call
        """
    if self._execing:
        raise RuntimeError('Nested set_execing_flag call')
    self._execing = True
    try:
        yield
    finally:
        self._execing = False