from _imp import (lock_held, acquire_lock, release_lock,
from importlib._bootstrap import _ERR_MSG, _exec, _load, _builtin_from_name
from importlib._bootstrap_external import SourcelessFileLoader
from importlib import machinery
from importlib import util
import importlib
import os
import sys
import tokenize
import types
import warnings
class NullImporter:
    """**DEPRECATED**

    Null import object.

    """

    def __init__(self, path):
        if path == '':
            raise ImportError('empty pathname', path='')
        elif os.path.isdir(path):
            raise ImportError('existing directory', path=path)

    def find_module(self, fullname):
        """Always returns None."""
        return None