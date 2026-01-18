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
def _get_unhandled_exception_frame(depth: int) -> Optional[FrameType]:
    try:
        return _thread_local_info.f_unhandled
    except:
        frame = _getframe(depth)
        f_unhandled = frame
        while f_unhandled is not None and f_unhandled.f_back is not None:
            f_back = f_unhandled.f_back
            filename = f_back.f_code.co_filename
            name = splitext(basename(filename))[0]
            if name == 'threading':
                if f_back.f_code.co_name in ('__bootstrap', '_bootstrap', '__bootstrap_inner', '_bootstrap_inner', 'run'):
                    break
            elif name == 'pydev_monkey':
                if f_back.f_code.co_name == '__call__':
                    break
            elif name == 'pydevd':
                if f_back.f_code.co_name in ('_exec', 'run', 'main'):
                    break
            elif name == 'pydevd_runpy':
                if f_back.f_code.co_name.startswith(('run', '_run')):
                    break
            f_unhandled = f_back
        if f_unhandled is not None:
            _thread_local_info.f_unhandled = f_unhandled
            return _thread_local_info.f_unhandled
        return f_unhandled