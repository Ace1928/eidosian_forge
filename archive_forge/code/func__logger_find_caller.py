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
def _logger_find_caller():
    code, frame = _get_caller(4)
    if code:
        return (code.co_filename, frame.f_lineno, code.co_name)
    else:
        return ('(unknown file)', 0, '(unknown function)')