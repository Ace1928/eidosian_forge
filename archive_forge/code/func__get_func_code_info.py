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
def _get_func_code_info(code_obj, frame_or_depth) -> FuncCodeInfo:
    """
    Provides code-object related info.

    Note that it contains informations on the breakpoints for a given function.
    If breakpoints change a new FuncCodeInfo instance will be created.

    Note that this can be called by any thread.
    """
    py_db = GlobalDebuggerHolder.global_dbg
    if py_db is None:
        return None
    func_code_info = _code_to_func_code_info_cache.get(code_obj)
    if func_code_info is not None:
        if func_code_info.pydb_mtime == py_db.mtime:
            return func_code_info
    cache_file_type: dict
    cache_file_type_key: tuple
    code = code_obj
    co_filename: str = code.co_filename
    co_name: str = code.co_name
    func_code_info = FuncCodeInfo()
    func_code_info.code_obj = code_obj
    code_line_info = _get_code_line_info(code_obj)
    line_to_offset = code_line_info.line_to_offset
    func_code_info.pydb_mtime = py_db.mtime
    func_code_info.co_filename = co_filename
    func_code_info.co_name = co_name
    try:
        abs_path_real_path_and_base = NORM_PATHS_AND_BASE_CONTAINER[co_filename]
    except:
        abs_path_real_path_and_base = get_abs_path_real_path_and_base_from_file(co_filename)
    func_code_info.abs_path_filename = abs_path_real_path_and_base[0]
    func_code_info.canonical_normalized_filename = abs_path_real_path_and_base[1]
    frame = None
    cache_file_type = py_db.get_cache_file_type()
    cache_file_type_key = (code.co_firstlineno, abs_path_real_path_and_base[0], code_obj)
    try:
        file_type = cache_file_type[cache_file_type_key]
    except:
        if frame is None:
            if frame_or_depth.__class__ == int:
                frame = _getframe(frame_or_depth + 1)
            else:
                frame = frame_or_depth
            assert frame.f_code is code_obj, '%s != %s' % (frame.f_code, code_obj)
        file_type = py_db.get_file_type(frame, abs_path_real_path_and_base)
    if file_type is not None:
        func_code_info.always_skip_code = True
        func_code_info.always_filtered_out = True
        _code_to_func_code_info_cache[code_obj] = func_code_info
        return func_code_info
    if pydevd_dont_trace.should_trace_hook is not None:
        if not pydevd_dont_trace.should_trace_hook(code_obj, func_code_info.abs_path_filename):
            if frame is None:
                if frame_or_depth.__class__ == int:
                    frame = _getframe(frame_or_depth + 1)
                else:
                    frame = frame_or_depth
            assert frame.f_code is code_obj
            func_code_info.always_filtered_out = True
            _code_to_func_code_info_cache[code_obj] = func_code_info
            return func_code_info
    if frame is None:
        if frame_or_depth.__class__ == int:
            frame = _getframe(frame_or_depth + 1)
        else:
            frame = frame_or_depth
        assert frame.f_code is code_obj
    func_code_info.filtered_out_force_checked = py_db.apply_files_filter(frame, func_code_info.abs_path_filename, True)
    if py_db.is_files_filter_enabled:
        func_code_info.always_filtered_out = func_code_info.filtered_out_force_checked
        if func_code_info.always_filtered_out:
            _code_to_func_code_info_cache[code_obj] = func_code_info
            return func_code_info
    else:
        func_code_info.always_filtered_out = False
    breakpoints: dict = py_db.breakpoints.get(func_code_info.canonical_normalized_filename)
    function_breakpoint: object = py_db.function_breakpoint_name_to_breakpoint.get(func_code_info.co_name)
    if function_breakpoint:
        func_code_info.function_breakpoint_found = True
        func_code_info.function_breakpoint = function_breakpoint
    if breakpoints:
        bp_line_to_breakpoint = {}
        for breakpoint_line, bp in breakpoints.items():
            if breakpoint_line in line_to_offset:
                bp_line_to_breakpoint[breakpoint_line] = bp
        func_code_info.breakpoint_found = bool(bp_line_to_breakpoint)
        func_code_info.bp_line_to_breakpoint = bp_line_to_breakpoint
    if py_db.plugin:
        plugin_manager = py_db.plugin
        is_tracked_frame = plugin_manager.is_tracked_frame(frame)
        if is_tracked_frame:
            if py_db.has_plugin_line_breaks:
                required_events_breakpoint = plugin_manager.required_events_breakpoint()
                func_code_info.plugin_line_breakpoint_found = 'line' in required_events_breakpoint
                func_code_info.plugin_call_breakpoint_found = 'call' in required_events_breakpoint
            required_events_stepping = plugin_manager.required_events_stepping()
            func_code_info.plugin_line_stepping: bool = 'line' in required_events_stepping
            func_code_info.plugin_call_stepping: bool = 'call' in required_events_stepping
            func_code_info.plugin_return_stepping: bool = 'return' in required_events_stepping
    _code_to_func_code_info_cache[code_obj] = func_code_info
    return func_code_info