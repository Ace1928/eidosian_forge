import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _run_after_forkers():
    items = list(_afterfork_registry.items())
    items.sort()
    for (index, ident, func), obj in items:
        try:
            func(obj)
        except Exception as e:
            info('after forker raised exception %s', e)