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
class TestCounterHooks(tests.TestCase, SelfTestHelper):
    _test_needs_features = [features.subunit]

    def setUp(self):
        super().setUp()

        class Test(tests.TestCase):

            def setUp(self):
                super().setUp()
                self.hooks = hooks.Hooks()
                self.hooks.add_hook('myhook', 'Foo bar blah', (2, 4))
                self.install_counter_hook(self.hooks, 'myhook')

            def no_hook(self):
                pass

            def run_hook_once(self):
                for hook in self.hooks['myhook']:
                    hook(self)
        self.test_class = Test

    def assertHookCalls(self, expected_calls, test_name):
        test = self.test_class(test_name)
        result = unittest.TestResult()
        test.run(result)
        self.assertTrue(hasattr(test, '_counters'))
        self.assertTrue('myhook' in test._counters)
        self.assertEqual(expected_calls, test._counters['myhook'])

    def test_no_hook(self):
        self.assertHookCalls(0, 'no_hook')

    def test_run_hook_once(self):
        tt = features.testtools
        if tt.module.__version__ < (0, 9, 8):
            raise tests.TestSkipped('testtools-0.9.8 required for addDetail')
        self.assertHookCalls(1, 'run_hook_once')