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
class TestInitSafetyIssues(TestCase):
    _DEBUG = False

    def run_cmd(self, cmdline):
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return (out.decode(), err.decode())

    @linux_only
    def test_orphaned_semaphore(self):
        test_file = os.path.join(os.path.dirname(__file__), 'orphaned_semaphore_usecase.py')
        cmdline = [sys.executable, test_file]
        out, err = self.run_cmd(cmdline)
        self.assertNotIn('leaked semaphore', err)
        if self._DEBUG:
            print('OUT:', out)
            print('ERR:', err)

    def test_lazy_lock_init(self):
        for meth in ('fork', 'spawn', 'forkserver'):
            try:
                multiprocessing.get_context(meth)
            except ValueError:
                continue
            cmd = "import numba; import multiprocessing;multiprocessing.set_start_method('{}');print(multiprocessing.get_context().get_start_method())"
            cmdline = [sys.executable, '-c', cmd.format(meth)]
            out, err = self.run_cmd(cmdline)
            if self._DEBUG:
                print('OUT:', out)
                print('ERR:', err)
            self.assertIn(meth, out)