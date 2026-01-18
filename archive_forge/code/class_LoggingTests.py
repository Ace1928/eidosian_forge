from __future__ import absolute_import
import threading
import warnings
import subprocess
import sys
from unittest import SkipTest, TestCase
import twisted
from twisted.python.log import PythonLoggingObserver
from twisted.python import log
from twisted.python.runtime import platform
from twisted.internet.task import Clock
from .._eventloop import EventLoop, ThreadLogObserver, _store
from ..tests import crochet_directory
import sys
import crochet
import sys
from logging import StreamHandler, Formatter, getLogger, DEBUG
import crochet
from twisted.python import log
from twisted.logger import Logger
import time
class LoggingTests(TestCase):
    """
    End-to-end tests for Twisted->stdlib logging bridge.
    """
    maxDiff = None

    def test_old_logging(self):
        """
        Messages from the old Twisted logging API are emitted to Python
        standard library logging.
        """
        if tuple(map(int, twisted.__version__.split('.'))) >= (15, 2, 0):
            raise SkipTest('This test is for Twisted < 15.2.')
        program = LOGGING_PROGRAM % ('',)
        output = subprocess.check_output([sys.executable, '-u', '-c', program], cwd=crochet_directory)
        self.assertTrue(output.startswith('INFO Log opened.\nINFO log-info\nERROR log-error\n'))

    def test_new_logging(self):
        """
        Messages from both new and old Twisted logging APIs are emitted to
        Python standard library logging.
        """
        if tuple(map(int, twisted.__version__.split('.'))) < (15, 2, 0):
            raise SkipTest('This test is for Twisted 15.2 and later.')
        program = LOGGING_PROGRAM % ('from twisted.logger import Logger\nl2 = Logger()\nimport time\ntime.sleep(1)  # workaround, there is race condition... somewhere\nl2.info("logger-info")\nl2.critical("logger-critical")\nl2.warn("logger-warning")\nl2.debug("logger-debug")\n',)
        output = subprocess.check_output([sys.executable, '-u', '-c', program], cwd=crochet_directory)
        self.assertIn('INFO logger-info\nCRITICAL logger-critical\nWARNING logger-warning\nDEBUG logger-debug\nINFO log-info\nCRITICAL log-error\n', output.decode('utf-8').replace('\r\n', '\n'))