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
def _create_thread_info(depth):
    thread_ident = _get_ident()
    f_bootstrap_frame, is_bootstrap_frame_internal = _get_bootstrap_frame(depth + 1)
    if f_bootstrap_frame is None:
        return None
    if is_bootstrap_frame_internal:
        t = None
        if f_bootstrap_frame.f_code.co_name in ('__bootstrap_inner', '_bootstrap_inner', 'is_alive'):
            t = f_bootstrap_frame.f_locals.get('self')
            if not isinstance(t, threading.Thread):
                t = None
        elif f_bootstrap_frame.f_code.co_name in ('_exec', '__call__'):
            t = f_bootstrap_frame.f_locals.get('t')
            if not isinstance(t, threading.Thread):
                t = None
    else:
        t = threading.current_thread()
    if t is None:
        t = _thread_active.get(thread_ident)
    if isinstance(t, threading._DummyThread):
        _thread_local_info._ref = _DeleteDummyThreadOnDel(t)
    if t is None:
        return None
    if getattr(t, 'is_pydev_daemon_thread', False):
        return ThreadInfo(t, thread_ident, False, None)
    else:
        try:
            additional_info = t.additional_info
            if additional_info is None:
                raise AttributeError()
        except:
            additional_info = set_additional_thread_info(t)
        return ThreadInfo(t, thread_ident, True, additional_info)