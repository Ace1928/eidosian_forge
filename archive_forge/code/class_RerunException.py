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
class RerunException(ScriptControlException):
    """Silently stop and rerun the user's script."""

    def __init__(self, rerun_data: RerunData):
        """Construct a RerunException

        Parameters
        ----------
        rerun_data : RerunData
            The RerunData that should be used to rerun the script
        """
        self.rerun_data = rerun_data

    def __repr__(self) -> str:
        return util.repr_(self)