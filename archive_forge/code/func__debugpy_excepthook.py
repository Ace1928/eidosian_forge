import logging
import os
import sys
import threading
import importlib
import ray
from ray.util.annotations import DeveloperAPI
def _debugpy_excepthook():
    """
    Drop the user into the debugger on an unhandled exception.
    """
    import threading
    import pydevd
    py_db = pydevd.get_global_debugger()
    thread = threading.current_thread()
    additional_info = py_db.set_additional_thread_info(thread)
    additional_info.is_tracing += 1
    try:
        error = sys.exc_info()
        py_db.stop_on_unhandled_exception(py_db, thread, additional_info, error)
        sys.excepthook(error[0], error[1], error[2])
    finally:
        additional_info.is_tracing -= 1