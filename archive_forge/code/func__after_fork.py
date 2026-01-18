import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
@staticmethod
def _after_fork():
    from . import util
    util._finalizer_registry.clear()
    util._run_after_forkers()