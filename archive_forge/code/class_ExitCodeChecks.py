import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
class ExitCodeChecks(tt.TempFileMixin):

    def setUp(self):
        self.system = ip.system_raw

    def test_exit_code_ok(self):
        self.system('exit 0')
        self.assertEqual(ip.user_ns['_exit_code'], 0)

    def test_exit_code_error(self):
        self.system('exit 1')
        self.assertEqual(ip.user_ns['_exit_code'], 1)

    @skipif(not hasattr(signal, 'SIGALRM'))
    def test_exit_code_signal(self):
        self.mktmp('import signal, time\nsignal.setitimer(signal.ITIMER_REAL, 0.1)\ntime.sleep(1)\n')
        self.system('%s %s' % (sys.executable, self.fname))
        self.assertEqual(ip.user_ns['_exit_code'], -signal.SIGALRM)

    @onlyif_cmds_exist('csh')
    def test_exit_code_signal_csh(self):
        SHELL = os.environ.get('SHELL', None)
        os.environ['SHELL'] = find_cmd('csh')
        try:
            self.test_exit_code_signal()
        finally:
            if SHELL is not None:
                os.environ['SHELL'] = SHELL
            else:
                del os.environ['SHELL']