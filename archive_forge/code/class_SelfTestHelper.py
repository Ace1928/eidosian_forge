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
class SelfTestHelper:

    def run_selftest(self, **kwargs):
        """Run selftest returning its output."""
        bio = BytesIO()
        output = TextIOWrapper(bio, 'utf-8')
        old_transport = breezy.tests.default_transport
        old_root = tests.TestCaseWithMemoryTransport.TEST_ROOT
        tests.TestCaseWithMemoryTransport.TEST_ROOT = None
        try:
            self.assertEqual(True, tests.selftest(stream=output, **kwargs))
        finally:
            breezy.tests.default_transport = old_transport
            tests.TestCaseWithMemoryTransport.TEST_ROOT = old_root
        output.flush()
        output.detach()
        bio.seek(0)
        return bio