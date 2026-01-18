from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
class DupHandle(object):
    """Picklable wrapper for a handle."""

    def __init__(self, handle, access, pid=None):
        if pid is None:
            pid = os.getpid()
        proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, pid)
        try:
            self._handle = _winapi.DuplicateHandle(_winapi.GetCurrentProcess(), handle, proc, access, False, 0)
        finally:
            _winapi.CloseHandle(proc)
        self._access = access
        self._pid = pid

    def detach(self):
        """Get the handle.  This should only be called once."""
        if self._pid == os.getpid():
            return self._handle
        proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, self._pid)
        try:
            return _winapi.DuplicateHandle(proc, self._handle, _winapi.GetCurrentProcess(), self._access, False, _winapi.DUPLICATE_CLOSE_SOURCE)
        finally:
            _winapi.CloseHandle(proc)