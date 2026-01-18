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
def _get_thread_info(create: bool, depth: int) -> Optional[ThreadInfo]:
    """
    Provides thread-related info.

    May return None if the thread is still not active.
    """
    try:
        return _thread_local_info.thread_info
    except:
        if not create:
            return None
        thread_info = _create_thread_info(depth + 1)
        if thread_info is None:
            return None
        _thread_local_info.thread_info = thread_info
        return _thread_local_info.thread_info