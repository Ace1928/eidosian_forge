import sys
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_comm import get_global_debugger
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
class DummyTracingHolder:
    dummy_trace_func = None

    def set_trace_func(self, trace_func):
        self.dummy_trace_func = trace_func