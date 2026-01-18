import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
class _ParentProcess(BaseProcess):

    def __init__(self, name, pid, sentinel):
        self._identity = ()
        self._name = name
        self._pid = pid
        self._parent_pid = None
        self._popen = None
        self._closed = False
        self._sentinel = sentinel
        self._config = {}

    def is_alive(self):
        from multiprocess.connection import wait
        return not wait([self._sentinel], timeout=0)

    @property
    def ident(self):
        return self._pid

    def join(self, timeout=None):
        """
        Wait until parent process terminates
        """
        from multiprocess.connection import wait
        wait([self._sentinel], timeout=timeout)
    pid = ident