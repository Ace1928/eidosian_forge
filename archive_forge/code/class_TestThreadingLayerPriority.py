import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
@skip_parfors_unsupported
class TestThreadingLayerPriority(ThreadLayerTestHelper):

    def each_env_var(self, env_var: str):
        """Test setting priority via env var NUMBA_THREADING_LAYER_PRIORITY.
        """
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'default'
        env['NUMBA_THREADING_LAYER_PRIORITY'] = env_var
        code = f'''\n                import numba\n\n                # trigger threading layer decision\n                # hence catching invalid THREADING_LAYER_PRIORITY\n                @numba.jit(\n                    'float64[::1](float64[::1], float64[::1])',\n                    nopython=True,\n                    parallel=True,\n                )\n                def plus(x, y):\n                    return x + y\n\n                captured_envvar = list("{env_var}".split())\n                assert numba.config.THREADING_LAYER_PRIORITY ==                     captured_envvar, "priority mismatch"\n                assert numba.threading_layer() == captured_envvar[0],                    "selected backend mismatch"\n                '''
        cmd = [sys.executable, '-c', textwrap.dedent(code)]
        self.run_cmd(cmd, env=env)

    @skip_no_omp
    @skip_no_tbb
    def test_valid_env_var(self):
        default = ['tbb', 'omp', 'workqueue']
        for p in itertools.permutations(default):
            env_var = ' '.join(p)
            self.each_env_var(env_var)

    @skip_no_omp
    @skip_no_tbb
    def test_invalid_env_var(self):
        env_var = 'tbb omp workqueue notvalidhere'
        with self.assertRaises(AssertionError) as raises:
            self.each_env_var(env_var)
        for msg in ('THREADING_LAYER_PRIORITY invalid:', 'It must be a permutation of'):
            self.assertIn(f'{msg}', str(raises.exception))

    @skip_no_omp
    def test_omp(self):
        for env_var in ('omp tbb workqueue', 'omp workqueue tbb'):
            self.each_env_var(env_var)

    @skip_no_tbb
    def test_tbb(self):
        for env_var in ('tbb omp workqueue', 'tbb workqueue omp'):
            self.each_env_var(env_var)

    def test_workqueue(self):
        for env_var in ('workqueue tbb omp', 'workqueue omp tbb'):
            self.each_env_var(env_var)