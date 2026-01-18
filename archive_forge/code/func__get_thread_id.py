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
def _get_thread_id():
    """Get id of current thread, suitable for logging as an unsigned quantity."""
    thread_id = _thread.get_ident()
    return thread_id & _THREAD_ID_MASK