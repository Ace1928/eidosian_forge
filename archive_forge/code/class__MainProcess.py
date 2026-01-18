import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
class _MainProcess(BaseProcess):

    def __init__(self):
        self._identity = ()
        self._name = 'MainProcess'
        self._parent_pid = None
        self._popen = None
        self._closed = False
        self._config = {'authkey': AuthenticationString(os.urandom(32)), 'semprefix': '/mp'}

    def close(self):
        pass