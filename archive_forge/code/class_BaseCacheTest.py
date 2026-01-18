import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
class BaseCacheTest(TestCase):
    usecases_file = None
    modname = None

    def setUp(self):
        self.tempdir = temp_directory('test_cache')
        sys.path.insert(0, self.tempdir)
        self.modfile = os.path.join(self.tempdir, self.modname + '.py')
        self.cache_dir = os.path.join(self.tempdir, '__pycache__')
        shutil.copy(self.usecases_file, self.modfile)
        os.chmod(self.modfile, stat.S_IREAD | stat.S_IWRITE)
        self.maxDiff = None

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.path.remove(self.tempdir)

    def import_module(self):
        old = sys.modules.pop(self.modname, None)
        if old is not None:
            cached = [old.__cached__]
            for fn in cached:
                try:
                    os.unlink(fn)
                except FileNotFoundError:
                    pass
        mod = import_dynamic(self.modname)
        self.assertEqual(mod.__file__.rstrip('co'), self.modfile)
        return mod

    def cache_contents(self):
        try:
            return [fn for fn in os.listdir(self.cache_dir) if not fn.endswith(('.pyc', '.pyo'))]
        except FileNotFoundError:
            return []

    def get_cache_mtimes(self):
        return dict(((fn, os.path.getmtime(os.path.join(self.cache_dir, fn))) for fn in sorted(self.cache_contents())))

    def check_pycache(self, n):
        c = self.cache_contents()
        self.assertEqual(len(c), n, c)

    def dummy_test(self):
        pass