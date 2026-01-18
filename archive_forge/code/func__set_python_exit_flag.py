import sys
import atexit
from torch._utils import ExceptionWrapper
from . import worker, signal_handling, pin_memory, collate, fetch
def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True