import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
def _get_caller(offset=3):
    """Returns a code and frame object for the lowest non-logging stack frame."""
    f = _sys._getframe(offset)
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return (code, f)
        f = f.f_back
    return (None, None)