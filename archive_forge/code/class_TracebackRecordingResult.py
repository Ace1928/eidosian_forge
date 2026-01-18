import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TracebackRecordingResult(tests.ExtendedTestResult):

    def __init__(self):
        tests.ExtendedTestResult.__init__(self, StringIO(), 0, 1)
        self.postcode = None

    def _post_mortem(self, tb=None):
        """Record the code object at the end of the current traceback"""
        tb = tb or sys.exc_info()[2]
        if tb is not None:
            next = tb.tb_next
            while next is not None:
                tb = next
                next = next.tb_next
            self.postcode = tb.tb_frame.f_code

    def report_error(self, test, err):
        pass

    def report_failure(self, test, err):
        pass