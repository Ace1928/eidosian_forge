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
class TestIsolatedEnv(tests.TestCase):
    """Test isolating tests from os.environ.

    Since we use tests that are already isolated from os.environ a bit of care
    should be taken when designing the tests to avoid bootstrap side-effects.
    The tests start an already clean os.environ which allow doing valid
    assertions about which variables are present or not and design tests around
    these assertions.
    """

    class ScratchMonkey(tests.TestCase):

        def test_me(self):
            pass

    def test_basics(self):
        self.assertTrue('BRZ_HOME' in tests.isolated_environ)
        self.assertEqual(None, tests.isolated_environ['BRZ_HOME'])
        self.assertFalse('BRZ_HOME' in os.environ)
        self.assertTrue('LINES' in tests.isolated_environ)
        self.assertEqual('25', tests.isolated_environ['LINES'])
        self.assertEqual('25', os.environ['LINES'])

    def test_injecting_unknown_variable(self):
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'BRZ_HOME': 'foo'})
        self.assertEqual('foo', os.environ['BRZ_HOME'])
        tests.restore_os_environ(test)
        self.assertFalse('BRZ_HOME' in os.environ)

    def test_injecting_known_variable(self):
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'LINES': '42'})
        self.assertEqual('42', os.environ['LINES'])
        tests.restore_os_environ(test)
        self.assertEqual('25', os.environ['LINES'])

    def test_deleting_variable(self):
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'LINES': None})
        self.assertTrue('LINES' not in os.environ)
        tests.restore_os_environ(test)
        self.assertEqual('25', os.environ['LINES'])