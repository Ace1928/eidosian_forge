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
class TestRunBzr(tests.TestCase):
    result = 0
    out = ''
    err = ''

    def _run_bzr_core(self, argv, encoding=None, stdin=None, stdout=None, stderr=None, working_dir=None):
        """Override _run_bzr_core to test how it is invoked by run_bzr.

        Attempts to run bzr from inside this class don't actually run it.

        We test how run_bzr actually invokes bzr in another location.  Here we
        only need to test that it passes the right parameters to run_bzr.
        """
        self.argv = list(argv)
        self.encoding = encoding
        self.stdin = stdin
        self.working_dir = working_dir
        stdout.write(self.out)
        stderr.write(self.err)
        return self.result

    def test_run_bzr_error(self):
        self.out = 'It sure does!\n'
        self.result = 34
        out, err = self.run_bzr_error(['^$'], ['rocks'], retcode=34)
        self.assertEqual(['rocks'], self.argv)
        self.assertEqual('It sure does!\n', out)
        self.assertEqual(out, self.out)
        self.assertEqual('', err)
        self.assertEqual(err, self.err)

    def test_run_bzr_error_regexes(self):
        self.out = ''
        self.err = 'bzr: ERROR: foobarbaz is not versioned'
        self.result = 3
        out, err = self.run_bzr_error(['bzr: ERROR: foobarbaz is not versioned'], ['file-id', 'foobarbaz'])

    def test_encoding(self):
        """Test that run_bzr passes encoding to _run_bzr_core"""
        self.run_bzr('foo bar')
        self.assertEqual(osutils.get_user_encoding(), self.encoding)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', encoding='baz')
        self.assertEqual('baz', self.encoding)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_stdin(self):
        self.run_bzr('foo bar', stdin='gam')
        self.assertEqual('gam', self.stdin)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', stdin='zippy')
        self.assertEqual('zippy', self.stdin)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_working_dir(self):
        """Test that run_bzr passes working_dir to _run_bzr_core"""
        self.run_bzr('foo bar')
        self.assertEqual(None, self.working_dir)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', working_dir='baz')
        self.assertEqual('baz', self.working_dir)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_reject_extra_keyword_arguments(self):
        self.assertRaises(TypeError, self.run_bzr, 'foo bar', error_regex=['error message'])