from collections import namedtuple
import dis
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from types import CodeType, FrameType
from typing import Dict, Optional, Tuple, Any
from os.path import basename, splitext
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (GlobalDebuggerHolder, ForkSafeLock,
from pydevd_file_utils import (NORM_PATHS_AND_BASE_CONTAINER,
from _pydevd_bundle.pydevd_trace_dispatch import should_stop_on_exception, handle_exception
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_HANDLED
from _pydevd_bundle.pydevd_trace_dispatch import is_unhandled_exception
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info, any_thread_stepping, PyDBAdditionalThreadInfo
def _enable_code_tracing_for_frame_and_parents(thread_info, frame):
    py_db: object = GlobalDebuggerHolder.global_dbg
    if py_db is None or py_db.pydb_disposed:
        return
    while frame is not None:
        func_code_info: FuncCodeInfo = _get_func_code_info(frame.f_code, frame)
        if func_code_info.always_skip_code:
            frame = frame.f_back
            continue
        _enable_code_tracing(py_db, thread_info.additional_info, func_code_info, frame.f_code, frame, False)
        frame = frame.f_back