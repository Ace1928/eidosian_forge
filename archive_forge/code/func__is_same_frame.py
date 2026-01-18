import linecache
import os.path
import re
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
import dis
def _is_same_frame(self, target_frame, current_frame):
    if target_frame is current_frame:
        return True
    info = self._args[2]
    if info.pydev_use_scoped_step_frame:
        if target_frame is not None and current_frame is not None:
            if target_frame.f_code.co_filename == current_frame.f_code.co_filename:
                f = current_frame.f_back
                if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                    f = f.f_back
                    if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                        return True
    return False