import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _remaining_time(self, endtime):
    """Convenience for _communicate when computing timeouts."""
    if endtime is None:
        return None
    else:
        return endtime - _time()