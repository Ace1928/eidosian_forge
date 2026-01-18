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
@skip_if_typeguard
class TestCacheFileCollision(unittest.TestCase):
    _numba_parallel_test_ = False
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_usecases.py')
    modname = 'caching_file_loc_fodder'
    source_text_1 = '\nfrom numba import njit\n@njit(cache=True)\ndef bar():\n    return 123\n'
    source_text_2 = '\nfrom numba import njit\n@njit(cache=True)\ndef bar():\n    return 321\n'

    def setUp(self):
        self.tempdir = temp_directory('test_cache_file_loc')
        sys.path.insert(0, self.tempdir)
        self.modname = 'module_name_that_is_unlikely'
        self.assertNotIn(self.modname, sys.modules)
        self.modname_bar1 = self.modname
        self.modname_bar2 = '.'.join([self.modname, 'foo'])
        foomod = os.path.join(self.tempdir, self.modname)
        os.mkdir(foomod)
        with open(os.path.join(foomod, '__init__.py'), 'w') as fout:
            print(self.source_text_1, file=fout)
        with open(os.path.join(foomod, 'foo.py'), 'w') as fout:
            print(self.source_text_2, file=fout)

    def tearDown(self):
        sys.modules.pop(self.modname_bar1, None)
        sys.modules.pop(self.modname_bar2, None)
        sys.path.remove(self.tempdir)

    def import_bar1(self):
        return import_dynamic(self.modname_bar1).bar

    def import_bar2(self):
        return import_dynamic(self.modname_bar2).bar

    def test_file_location(self):
        bar1 = self.import_bar1()
        bar2 = self.import_bar2()
        idxname1 = bar1._cache._cache_file._index_name
        idxname2 = bar2._cache._cache_file._index_name
        self.assertNotEqual(idxname1, idxname2)
        self.assertTrue(idxname1.startswith('__init__.bar-3.py'))
        self.assertTrue(idxname2.startswith('foo.bar-3.py'))

    @unittest.skipUnless(hasattr(multiprocessing, 'get_context'), 'Test requires multiprocessing.get_context')
    def test_no_collision(self):
        bar1 = self.import_bar1()
        bar2 = self.import_bar2()
        with capture_cache_log() as buf:
            res1 = bar1()
        cachelog = buf.getvalue()
        self.assertEqual(cachelog.count('index saved'), 1)
        self.assertEqual(cachelog.count('data saved'), 1)
        self.assertEqual(cachelog.count('index loaded'), 0)
        self.assertEqual(cachelog.count('data loaded'), 0)
        with capture_cache_log() as buf:
            res2 = bar2()
        cachelog = buf.getvalue()
        self.assertEqual(cachelog.count('index saved'), 1)
        self.assertEqual(cachelog.count('data saved'), 1)
        self.assertEqual(cachelog.count('index loaded'), 0)
        self.assertEqual(cachelog.count('data loaded'), 0)
        self.assertNotEqual(res1, res2)
        try:
            mp = multiprocessing.get_context('spawn')
        except ValueError:
            print('missing spawn context')
        q = mp.Queue()
        proc = mp.Process(target=cache_file_collision_tester, args=(q, self.tempdir, self.modname_bar1, self.modname_bar2))
        proc.start()
        log1 = q.get()
        got1 = q.get()
        log2 = q.get()
        got2 = q.get()
        proc.join()
        self.assertEqual(got1, res1)
        self.assertEqual(got2, res2)
        self.assertEqual(log1.count('index saved'), 0)
        self.assertEqual(log1.count('data saved'), 0)
        self.assertEqual(log1.count('index loaded'), 1)
        self.assertEqual(log1.count('data loaded'), 1)
        self.assertEqual(log2.count('index saved'), 0)
        self.assertEqual(log2.count('data saved'), 0)
        self.assertEqual(log2.count('index loaded'), 1)
        self.assertEqual(log2.count('data loaded'), 1)