import unittest
import sys
import os
from io import BytesIO
from distutils import cygwinccompiler
from distutils.cygwinccompiler import (check_config_h,
from distutils.tests import support
class FakePopen(object):
    test_class = None

    def __init__(self, cmd, shell, stdout):
        self.cmd = cmd.split()[0]
        exes = self.test_class._exes
        if self.cmd in exes:
            self.stdout = BytesIO(exes[self.cmd])
        else:
            self.stdout = os.popen(cmd, 'r')