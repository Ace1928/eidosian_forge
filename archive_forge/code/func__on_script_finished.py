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
def _on_script_finished(self, ctx: ScriptRunContext, event: ScriptRunnerEvent, premature_stop: bool) -> None:
    """Called when our script finishes executing, even if it finished
        early with an exception. We perform post-run cleanup here.
        """
    if not premature_stop:
        self._session_state.on_script_finished(ctx.widget_ids_this_run)
    self.on_event.send(self, event=event)
    runtime.get_instance().media_file_mgr.remove_orphaned_files()
    if config.get_option('runner.postScriptGC'):
        gc.collect(2)