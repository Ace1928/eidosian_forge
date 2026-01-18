import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
class LockutilsModuleTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(LockutilsModuleTestCase, self).setUp()
        self.old_env = os.environ.get('OSLO_LOCK_PATH')
        if self.old_env is not None:
            del os.environ['OSLO_LOCK_PATH']

    def tearDown(self):
        if self.old_env is not None:
            os.environ['OSLO_LOCK_PATH'] = self.old_env
        super(LockutilsModuleTestCase, self).tearDown()

    def test_main(self):
        script = '\n'.join(['import os', 'lock_path = os.environ.get("OSLO_LOCK_PATH")', 'assert lock_path is not None', 'assert os.path.isdir(lock_path)'])
        argv = ['', sys.executable, '-c', script]
        retval = lockutils._lock_wrapper(argv)
        self.assertEqual(0, retval, 'Bad OSLO_LOCK_PATH has been set')

    def test_return_value_maintained(self):
        script = '\n'.join(['import sys', 'sys.exit(1)'])
        argv = ['', sys.executable, '-c', script]
        retval = lockutils._lock_wrapper(argv)
        self.assertEqual(1, retval)

    def test_direct_call_explodes(self):
        cmd = [sys.executable, '-m', 'oslo_concurrency.lockutils']
        with open(os.devnull, 'w') as devnull:
            retval = subprocess.call(cmd, stderr=devnull)
            self.assertEqual(1, retval)