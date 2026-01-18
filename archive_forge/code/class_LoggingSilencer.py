import os
import sys
import shutil
import tempfile
import unittest
import sysconfig
from copy import deepcopy
from test.support import os_helper
from distutils import log
from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
from distutils.core import Distribution
class LoggingSilencer(object):

    def setUp(self):
        super().setUp()
        self.threshold = log.set_threshold(log.FATAL)
        self._old_log = log.Log._log
        log.Log._log = self._log
        self.logs = []

    def tearDown(self):
        log.set_threshold(self.threshold)
        log.Log._log = self._old_log
        super().tearDown()

    def _log(self, level, msg, args):
        if level not in (DEBUG, INFO, WARN, ERROR, FATAL):
            raise ValueError('%s wrong log level' % str(level))
        if not isinstance(msg, str):
            raise TypeError("msg should be str, not '%.200s'" % type(msg).__name__)
        self.logs.append((level, msg, args))

    def get_logs(self, *levels):
        return [msg % args for level, msg, args in self.logs if level in levels]

    def clear_logs(self):
        self.logs = []