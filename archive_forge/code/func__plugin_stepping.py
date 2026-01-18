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
def _plugin_stepping(py_db, step_cmd, event, frame, thread_info):
    plugin_manager = py_db.plugin
    if step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE, CMD_SMART_STEP_INTO) or step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
        stop_info = {}
        stop = False
        result = plugin_manager.cmd_step_into(py_db, frame, event, thread_info.additional_info, thread_info.thread, stop_info, stop)
        if result:
            stop, plugin_stop = result
            if plugin_stop:
                plugin_manager.stop(py_db, frame, event, thread_info.thread, stop_info, None, step_cmd)
                return
    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE):
        if plugin_manager is not None:
            stop_info = {}
            stop = False
            result = plugin_manager.cmd_step_over(py_db, frame, event, thread_info.additional_info, thread_info.thread, stop_info, stop)
            if result:
                stop, plugin_stop = result
                if plugin_stop:
                    plugin_manager.stop(py_db, frame, event, thread_info.thread, stop_info, None, step_cmd)
                    return