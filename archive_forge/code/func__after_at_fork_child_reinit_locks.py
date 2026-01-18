import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _after_at_fork_child_reinit_locks():
    for handler in _at_fork_reinit_lock_weakset:
        handler._at_fork_reinit()
    _lock._at_fork_reinit()