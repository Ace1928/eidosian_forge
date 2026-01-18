from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
@staticmethod
def __recycle_threads():

    def worker():
        time.sleep(0.001)
    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.001)
    t.join(10)