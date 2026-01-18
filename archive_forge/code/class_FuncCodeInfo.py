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
class FuncCodeInfo:

    def __init__(self):
        self.co_filename: str = ''
        self.canonical_normalized_filename: str = ''
        self.abs_path_filename: str = ''
        self.always_skip_code: bool = False
        self.breakpoint_found: bool = False
        self.function_breakpoint_found: bool = False
        self.plugin_line_breakpoint_found: bool = False
        self.plugin_call_breakpoint_found: bool = False
        self.plugin_line_stepping: bool = False
        self.plugin_call_stepping: bool = False
        self.plugin_return_stepping: bool = False
        self.pydb_mtime: int = -1
        self.bp_line_to_breakpoint: Dict[int, Any] = {}
        self.function_breakpoint = None
        self.always_filtered_out: bool = False
        self.filtered_out_force_checked: bool = False
        self.try_except_container_obj: Optional[_TryExceptContainerObj] = None
        self.code_obj: CodeType = None
        self.co_name: str = ''

    def get_line_of_offset(self, offset):
        for start, end, line in self.code_obj.co_lines():
            if offset >= start and offset <= end:
                return line
        return -1