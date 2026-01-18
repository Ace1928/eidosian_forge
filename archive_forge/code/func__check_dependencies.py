import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def _check_dependencies(self, executables=(), modules=(), disable_viewers=(), python_version=(3, 5)):
    """
        Checks if the dependencies for the test are installed.

        Raises ``DependencyError`` it at least one dependency is not installed.
        """
    for executable in executables:
        if not shutil.which(executable):
            raise DependencyError('Could not find %s' % executable)
    for module in modules:
        if module == 'matplotlib':
            matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']}, min_module_version='1.0.0', catch=(RuntimeError,))
            if matplotlib is None:
                raise DependencyError('Could not import matplotlib')
        elif not import_module(module):
            raise DependencyError('Could not import %s' % module)
    if disable_viewers:
        tempdir = tempfile.mkdtemp()
        os.environ['PATH'] = '%s:%s' % (tempdir, os.environ['PATH'])
        vw = '#!/usr/bin/env python3\nimport sys\nif len(sys.argv) <= 1:\n    exit("wrong number of args")\n'
        for viewer in disable_viewers:
            with open(os.path.join(tempdir, viewer), 'w') as fh:
                fh.write(vw)
            os.chmod(os.path.join(tempdir, viewer), stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
    if python_version:
        if sys.version_info < python_version:
            raise DependencyError('Requires Python >= ' + '.'.join(map(str, python_version)))
    if 'pyglet' in modules:
        import pyglet

        class DummyWindow:

            def __init__(self, *args, **kwargs):
                self.has_exit = True
                self.width = 600
                self.height = 400

            def set_vsync(self, x):
                pass

            def switch_to(self):
                pass

            def push_handlers(self, x):
                pass

            def close(self):
                pass
        pyglet.window.Window = DummyWindow