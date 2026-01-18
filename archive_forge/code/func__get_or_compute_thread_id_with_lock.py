from __future__ import nested_scopes
import platform
import weakref
import struct
import warnings
import functools
from contextlib import contextmanager
import sys  # Note: the sys import must be here anyways (others depend on it)
import codecs as _codecs
import os
from _pydevd_bundle import pydevd_vm_type
from _pydev_bundle._pydev_saved_modules import thread, threading
def _get_or_compute_thread_id_with_lock(thread, is_current_thread):
    with _thread_id_lock:
        tid = getattr(thread, '__pydevd_id__', None)
        if tid is not None:
            return tid
        _thread_id_to_thread_found[id(thread)] = thread
        pid = get_pid()
        tid = 'pid_%s_id_%s' % (pid, id(thread))
        thread.__pydevd_id__ = tid
    return tid