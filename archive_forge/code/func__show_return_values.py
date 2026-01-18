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
def _show_return_values(self, frame, arg):
    try:
        try:
            f_locals_back = getattr(frame.f_back, 'f_locals', None)
            if f_locals_back is not None:
                return_values_dict = f_locals_back.get(RETURN_VALUES_DICT, None)
                if return_values_dict is None:
                    return_values_dict = {}
                    f_locals_back[RETURN_VALUES_DICT] = return_values_dict
                name = self.get_func_name(frame)
                return_values_dict[name] = arg
        except:
            pydev_log.exception()
    finally:
        f_locals_back = None