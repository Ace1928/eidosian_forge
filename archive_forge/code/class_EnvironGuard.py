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
class EnvironGuard(object):

    def setUp(self):
        super(EnvironGuard, self).setUp()
        self.old_environ = deepcopy(os.environ)

    def tearDown(self):
        for key, value in self.old_environ.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
        for key in tuple(os.environ.keys()):
            if key not in self.old_environ:
                del os.environ[key]
        super(EnvironGuard, self).tearDown()