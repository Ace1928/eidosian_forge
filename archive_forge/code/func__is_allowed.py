import contextlib
import threading
import warnings
def _is_allowed():
    try:
        return _thread_local.allowed
    except AttributeError:
        _thread_local.allowed = True
        return True