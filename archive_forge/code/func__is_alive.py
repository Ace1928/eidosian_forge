import os
import sys
import signal
import itertools
import logging
import threading
from _weakrefset import WeakSet
from multiprocessing import process as _mproc
def _is_alive(self):
    if self._popen is None:
        return False
    return self._popen.poll() is None