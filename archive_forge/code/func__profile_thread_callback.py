import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _profile_thread_callback(frame, event, arg):
    """
    _profile_thread_callback will only be called once per-thread. _yappi will detect
    the new thread and changes the profilefunc param of the ThreadState
    structure. This is an internal function please don't mess with it.
    """
    _yappi._profile_event(frame, event, arg)