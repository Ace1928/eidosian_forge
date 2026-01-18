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
class TestParallelFork(_ForkedSelftest, tests.TestCase):
    """Check operation of --parallel=fork selftest option"""

    def test_error_in_child_during_fork(self):
        """Error in a forked child during test setup should get reported"""

        class Test(tests.TestCase):

            def testMethod(self):
                pass
        self.overrideAttr(tests, 'workaround_zealous_crypto_random', None)
        out = self._run_selftest(test_suite_factory=Test)
        self.assertContainsRe(out, b'Traceback.*:\n(?:.*\n)*.+ in fork_for_tests\n(?:.*\n)*\\s*workaround_zealous_crypto_random\\(\\)\n(?:.*\n)*TypeError:')