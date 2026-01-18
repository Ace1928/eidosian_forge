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
def _enable_step_tracing(py_db, code, step_cmd, info, frame):
    if step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE, CMD_SMART_STEP_INTO):
        _enable_line_tracing(code)
        _enable_return_tracing(code)
    elif step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE) and _is_same_frame(info, info.pydev_step_stop, frame):
        _enable_return_tracing(code)
    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE):
        if _is_same_frame(info, info.pydev_step_stop, frame):
            _enable_line_tracing(code)
            _enable_return_tracing(code)
        elif py_db.show_return_values and _is_same_frame(info, info.pydev_step_stop, frame.f_back):
            _enable_return_tracing(code)