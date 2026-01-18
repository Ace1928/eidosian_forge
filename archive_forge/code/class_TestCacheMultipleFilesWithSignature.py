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
class TestCacheMultipleFilesWithSignature(unittest.TestCase):
    _numba_parallel_test_ = False
    source_text_file1 = '\nfrom file2 import function2\n'
    source_text_file2 = "\nfrom numba import njit\n\n@njit('float64(float64)', cache=True)\ndef function1(x):\n    return x\n\n@njit('float64(float64)', cache=True)\ndef function2(x):\n    return x\n"

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_loc')
        self.file1 = os.path.join(self.tempdir, 'file1.py')
        with open(self.file1, 'w') as fout:
            print(self.source_text_file1, file=fout)
        self.file2 = os.path.join(self.tempdir, 'file2.py')
        with open(self.file2, 'w') as fout:
            print(self.source_text_file2, file=fout)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_caching_mutliple_files_with_signature(self):
        popen = subprocess.Popen([sys.executable, self.file1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f'stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}'
        self.assertEqual(popen.returncode, 0, msg=msg)
        popen = subprocess.Popen([sys.executable, self.file2], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f'stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}'
        self.assertEqual(popen.returncode, 0, msg)