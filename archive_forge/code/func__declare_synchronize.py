import contextlib
import threading
import warnings
def _declare_synchronize():
    if not _is_allowed():
        raise DeviceSynchronized()