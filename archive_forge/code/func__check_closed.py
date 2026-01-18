import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
def _check_closed(self):
    if self._closed:
        raise ValueError('process object is closed')