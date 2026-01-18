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
class TestRunBzrSubprocess(TestWithFakedStartBzrSubprocess):

    def assertRunBzrSubprocess(self, expected_args, process, *args, **kwargs):
        """Run run_brz_subprocess with args and kwargs using a stubbed process.

        Inside TestRunBzrSubprocessCommands we use a stub start_brz_subprocess
        that will return static results. This assertion method populates those
        results and also checks the arguments run_brz_subprocess generates.
        """
        self.next_subprocess = process
        try:
            result = self.run_brz_subprocess(*args, **kwargs)
        except BaseException:
            self.next_subprocess = None
            for key, expected in expected_args.items():
                self.assertEqual(expected, self.subprocess_calls[-1][key])
            raise
        else:
            self.next_subprocess = None
            for key, expected in expected_args.items():
                self.assertEqual(expected, self.subprocess_calls[-1][key])
            return result

    def test_run_brz_subprocess(self):
        """The run_bzr_helper_external command behaves nicely."""
        self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), '--version')
        self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), ['--version'])
        result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--version', retcode=None)
        result = self.assertRunBzrSubprocess({}, StubProcess(out='is free software'), '--version')
        self.assertContainsRe(result[0], 'is free software')
        self.assertRaises(AssertionError, self.assertRunBzrSubprocess, {'process_args': ['--versionn']}, StubProcess(retcode=3), '--versionn')
        result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--versionn', retcode=3)
        result = self.assertRunBzrSubprocess({}, StubProcess(err='unknown command', retcode=3), '--versionn', retcode=None)
        self.assertContainsRe(result[1], 'unknown command')

    def test_env_change_passes_through(self):
        self.assertRunBzrSubprocess({'env_changes': {'new': 'value', 'changed': 'newvalue', 'deleted': None}}, StubProcess(), '', env_changes={'new': 'value', 'changed': 'newvalue', 'deleted': None})

    def test_no_working_dir_passed_as_None(self):
        self.assertRunBzrSubprocess({'working_dir': None}, StubProcess(), '')

    def test_no_working_dir_passed_through(self):
        self.assertRunBzrSubprocess({'working_dir': 'dir'}, StubProcess(), '', working_dir='dir')

    def test_run_brz_subprocess_no_plugins(self):
        self.assertRunBzrSubprocess({'allow_plugins': False}, StubProcess(), '')

    def test_allow_plugins(self):
        self.assertRunBzrSubprocess({'allow_plugins': True}, StubProcess(), '', allow_plugins=True)