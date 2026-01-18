import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
def active_children():
    """
    Return list of process objects corresponding to live child processes
    """
    _cleanup()
    return list(_children)