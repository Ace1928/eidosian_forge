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
def _maybe_handle_execution_control_request(self) -> None:
    """Check our current ScriptRequestState to see if we have a
        pending STOP or RERUN request.

        This function is called every time the app script enqueues a
        ForwardMsg, which means that most `st.foo` commands - which generally
        involve sending a ForwardMsg to the frontend - act as implicit
        yield points in the script's execution.
        """
    if not self._is_in_script_thread():
        return
    if not self._execing:
        return
    request = self._requests.on_scriptrunner_yield()
    if request is None:
        return
    if request.type == ScriptRequestType.RERUN:
        raise RerunException(request.rerun_data)
    assert request.type == ScriptRequestType.STOP
    raise StopException()