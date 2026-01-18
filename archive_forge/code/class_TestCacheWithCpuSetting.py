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
class TestCacheWithCpuSetting(DispatcherCacheUsecasesTest):
    _numba_parallel_test_ = False

    def check_later_mtimes(self, mtimes_old):
        match_count = 0
        for k, v in self.get_cache_mtimes().items():
            if k in mtimes_old:
                self.assertGreaterEqual(v, mtimes_old[k])
                match_count += 1
        self.assertGreater(match_count, 0, msg='nothing to compare')

    def test_user_set_cpu_name(self):
        self.check_pycache(0)
        mod = self.import_module()
        mod.self_test()
        cache_size = len(self.cache_contents())
        mtimes = self.get_cache_mtimes()
        self.run_in_separate_process(envvars={'NUMBA_CPU_NAME': 'generic'})
        self.check_later_mtimes(mtimes)
        self.assertGreater(len(self.cache_contents()), cache_size)
        cache = mod.add_usecase._cache
        cache_file = cache._cache_file
        cache_index = cache_file._load_index()
        self.assertEqual(len(cache_index), 2)
        [key_a, key_b] = cache_index.keys()
        if key_a[1][1] == ll.get_host_cpu_name():
            key_host, key_generic = (key_a, key_b)
        else:
            key_host, key_generic = (key_b, key_a)
        self.assertEqual(key_host[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_host[1][2], codegen.get_host_cpu_features())
        self.assertEqual(key_generic[1][1], 'generic')
        self.assertEqual(key_generic[1][2], '')

    def test_user_set_cpu_features(self):
        self.check_pycache(0)
        mod = self.import_module()
        mod.self_test()
        cache_size = len(self.cache_contents())
        mtimes = self.get_cache_mtimes()
        my_cpu_features = '-sse;-avx'
        system_features = codegen.get_host_cpu_features()
        self.assertNotEqual(system_features, my_cpu_features)
        self.run_in_separate_process(envvars={'NUMBA_CPU_FEATURES': my_cpu_features})
        self.check_later_mtimes(mtimes)
        self.assertGreater(len(self.cache_contents()), cache_size)
        cache = mod.add_usecase._cache
        cache_file = cache._cache_file
        cache_index = cache_file._load_index()
        self.assertEqual(len(cache_index), 2)
        [key_a, key_b] = cache_index.keys()
        if key_a[1][2] == system_features:
            key_host, key_generic = (key_a, key_b)
        else:
            key_host, key_generic = (key_b, key_a)
        self.assertEqual(key_host[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_host[1][2], system_features)
        self.assertEqual(key_generic[1][1], ll.get_host_cpu_name())
        self.assertEqual(key_generic[1][2], my_cpu_features)