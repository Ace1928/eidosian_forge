from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
class ColouredResult(unittest.TextTestResult):

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        cprint('OK', 'green')

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        cprint('ERROR', 'red', bold=True)

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        cprint('FAIL', 'red')

    def addSkip(self, test, reason):
        unittest.TestResult.addSkip(self, test, reason)
        cprint('skipped: %s' % reason.strip(), 'brown')

    def printErrorList(self, flavour, errors):
        flavour = hilite(flavour, 'red', bold=flavour == 'ERROR')
        super().printErrorList(flavour, errors)