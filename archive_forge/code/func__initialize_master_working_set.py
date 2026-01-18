import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
@_call_aside
def _initialize_master_working_set():
    """
    Prepare the master working set and make the ``require()``
    API available.

    This function has explicit effects on the global state
    of pkg_resources. It is intended to be invoked once at
    the initialization of this module.

    Invocation by other packages is unsupported and done
    at their own risk.
    """
    working_set = WorkingSet._build_master()
    _declare_state('object', working_set=working_set)
    require = working_set.require
    iter_entry_points = working_set.iter_entry_points
    add_activation_listener = working_set.subscribe
    run_script = working_set.run_script
    run_main = run_script
    tuple((dist.activate(replace=False) for dist in working_set))
    add_activation_listener(lambda dist: dist.activate(replace=True), existing=False)
    working_set.entries = []
    list(map(working_set.add_entry, sys.path))
    globals().update(locals())