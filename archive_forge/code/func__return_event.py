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
def _return_event(code, instruction, retval):
    try:
        thread_info = _thread_local_info.thread_info
    except:
        thread_info = _get_thread_info(True, 1)
        if thread_info is None:
            return
    py_db: object = GlobalDebuggerHolder.global_dbg
    if py_db is None or py_db.pydb_disposed:
        return monitor.DISABLE
    if not thread_info.trace or thread_info.thread._is_stopped:
        return
    func_code_info: FuncCodeInfo = _get_func_code_info(code, 1)
    if func_code_info.always_skip_code:
        return monitor.DISABLE
    info = thread_info.additional_info
    frame = _getframe(1)
    step_cmd = info.pydev_step_cmd
    if step_cmd == -1:
        return
    if info.suspend_type != PYTHON_SUSPEND:
        if func_code_info.plugin_return_stepping:
            _plugin_stepping(py_db, step_cmd, 'return', frame, thread_info)
        return
    stop_frame = info.pydev_step_stop
    if step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE):
        force_check_project_scope = step_cmd == CMD_STEP_INTO_MY_CODE
        if frame.f_back is not None and (not info.pydev_use_scoped_step_frame):
            back_func_code_info = _get_func_code_info(frame.f_back.f_code, frame.f_back)
            if not back_func_code_info.always_skip_code and (not back_func_code_info.always_filtered_out) and (not (force_check_project_scope and back_func_code_info.filtered_out_force_checked)) and (info.step_in_initial_location != (frame.f_back, frame.f_back.f_lineno)):
                if py_db.show_return_values:
                    _show_return_values(frame, retval)
                _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval)
                return
    if step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE) and _is_same_frame(info, stop_frame, frame):
        if py_db.show_return_values:
            _show_return_values(frame, retval)
        _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval)
        return
    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE) and (not info.pydev_use_scoped_step_frame) and _is_same_frame(info, stop_frame, frame):
        f_back = frame.f_back
        if f_back is not None:
            back_func_code_info = _get_func_code_info(f_back.f_code, 2)
            force_check_project_scope = step_cmd == CMD_STEP_OVER_MY_CODE
            if back_func_code_info is not None and (not back_func_code_info.always_skip_code) and (not back_func_code_info.always_filtered_out) and (not (force_check_project_scope and back_func_code_info.filtered_out_force_checked)):
                if py_db.show_return_values:
                    _show_return_values(frame, retval)
                _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval)
                return
    elif step_cmd == CMD_SMART_STEP_INTO:
        if _is_same_frame(info, stop_frame, frame):
            if py_db.show_return_values:
                _show_return_values(frame, retval)
            _stop_on_return(py_db, thread_info, info, step_cmd, frame, retval)
            return
    if py_db.show_return_values:
        if info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO) and _is_same_frame(info, stop_frame, frame.f_back) or (info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE) and (info, _is_same_frame(info, stop_frame, frame))) or info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_COROUTINE) or (info.pydev_step_cmd == CMD_STEP_INTO_MY_CODE and frame.f_back is not None and (not py_db.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True))):
            _show_return_values(frame, retval)
    if step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_STEP_OVER_MY_CODE, CMD_STEP_RETURN_MY_CODE, CMD_SMART_STEP_INTO):
        stop_frame = info.pydev_step_stop
        if stop_frame is frame and (not info.pydev_use_scoped_step_frame):
            if step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_SMART_STEP_INTO):
                info.pydev_step_cmd = CMD_STEP_INTO
            else:
                info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE
            info.pydev_step_stop = None
            _enable_code_tracing_for_frame_and_parents(thread_info, stop_frame.f_back)
            if py_db.show_return_values:
                _show_return_values(frame, retval)