import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _platform_supports_abstract_sockets():
    if sys.platform == 'linux':
        return True
    if hasattr(sys, 'getandroidapilevel'):
        return True
    return False