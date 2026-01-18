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
def _filter_handle_list(self, handle_list):
    """Filter out console handles that can't be used
            in lpAttributeList["handle_list"] and make sure the list
            isn't empty. This also removes duplicate handles."""
    return list({handle for handle in handle_list if handle & 3 != 3 or _winapi.GetFileType(handle) != _winapi.FILE_TYPE_CHAR})