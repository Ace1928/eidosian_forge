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
def _run_script_thread(self) -> None:
    """The entry point for the script thread.

        Processes the ScriptRequestQueue, which will at least contain the RERUN
        request that will trigger the first script-run.

        When the ScriptRequestQueue is empty, or when a SHUTDOWN request is
        dequeued, this function will exit and its thread will terminate.
        """
    assert self._is_in_script_thread()
    _LOGGER.debug('Beginning script thread')
    ctx = ScriptRunContext(session_id=self._session_id, _enqueue=self._enqueue_forward_msg, script_requests=self._requests, query_string='', session_state=self._session_state, uploaded_file_mgr=self._uploaded_file_mgr, main_script_path=self._main_script_path, page_script_hash='', user_info=self._user_info, gather_usage_stats=bool(config.get_option('browser.gatherUsageStats')), fragment_storage=self._fragment_storage)
    add_script_run_ctx(threading.current_thread(), ctx)
    request = self._requests.on_scriptrunner_ready()
    while request.type == ScriptRequestType.RERUN:
        self._run_script(request.rerun_data)
        request = self._requests.on_scriptrunner_ready()
    assert request.type == ScriptRequestType.STOP
    client_state = ClientState()
    client_state.query_string = ctx.query_string
    client_state.page_script_hash = ctx.page_script_hash
    self.on_event.send(self, event=ScriptRunnerEvent.SHUTDOWN, client_state=client_state)