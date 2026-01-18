import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def _get_revised_path(given_path, argv0):
    """Ensures current directory is on returned path, and argv0 directory is not

    Exception: argv0 dir is left alone if it's also pydoc's directory.

    Returns a new path entry list, or None if no adjustment is needed.
    """
    if '' in given_path or os.curdir in given_path or os.getcwd() in given_path:
        return None
    stdlib_dir = os.path.dirname(__file__)
    script_dir = os.path.dirname(argv0)
    revised_path = given_path.copy()
    if script_dir in given_path and (not os.path.samefile(script_dir, stdlib_dir)):
        revised_path.remove(script_dir)
    revised_path.insert(0, os.getcwd())
    return revised_path