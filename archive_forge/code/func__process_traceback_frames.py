import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def _process_traceback_frames(tb):
    new_tb = None
    tb_list = list(traceback.walk_tb(tb))
    for f, line_no in reversed(tb_list):
        if include_frame(f.f_code.co_filename):
            new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
    if new_tb is None and tb_list:
        f, line_no = tb_list[-1]
        new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
    return new_tb