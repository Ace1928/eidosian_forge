import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _at_fork_reinit(self):
    self._lock._at_fork_reinit()