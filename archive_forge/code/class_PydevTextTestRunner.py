import unittest as python_unittest
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydevd_bundle import pydevd_io
import traceback
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
from io import StringIO
class PydevTextTestRunner(python_unittest.TextTestRunner):

    def _makeResult(self):
        return PydevTestResult(self.stream, self.descriptions, self.verbosity)