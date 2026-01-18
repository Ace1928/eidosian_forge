from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, NO_FTRACE,
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_frame import PyDBFrame, is_unhandled_exception
class TopLevelThreadTracerOnlyUnhandledExceptions(object):

    def __init__(self, args):
        self._args = args

    def trace_unhandled_exceptions(self, frame, event, arg):
        if event == 'exception' and arg is not None:
            py_db, t, additional_info = self._args[0:3]
            if arg is not None:
                if not additional_info.suspended_at_unhandled:
                    additional_info.suspended_at_unhandled = True
                    py_db.stop_on_unhandled_exception(py_db, t, additional_info, arg)
        return self.trace_unhandled_exceptions

    def get_trace_dispatch_func(self):
        return self.trace_unhandled_exceptions