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
def _get_script_run_ctx(self) -> ScriptRunContext:
    """Get the ScriptRunContext for the current thread.

        Returns
        -------
        ScriptRunContext
            The ScriptRunContext for the current thread.

        Raises
        ------
        AssertionError
            If called outside of a ScriptRunner thread.
        RuntimeError
            If there is no ScriptRunContext for the current thread.

        """
    assert self._is_in_script_thread()
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError('ScriptRunner thread has a null ScriptRunContext. Something has gone very wrong!')
    return ctx