import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
class VerboseTestResult(ExtendedTestResult):
    """Produce long output, with one line per test run plus times"""

    def _ellipsize_to_right(self, a_string, final_width):
        """Truncate and pad a string, keeping the right hand side"""
        if len(a_string) > final_width:
            result = '...' + a_string[3 - final_width:]
        else:
            result = a_string
        return result.ljust(final_width)

    def report_tests_starting(self):
        self.stream.write('running %d tests...\n' % self.num_tests)
        super().report_tests_starting()

    def report_test_start(self, test):
        name = self._shortened_test_description(test)
        width = osutils.terminal_width()
        if width is not None:
            self.stream.write(self._ellipsize_to_right(name, width - 18))
        else:
            self.stream.write(name)
        self.stream.flush()

    def _error_summary(self, err):
        indent = ' ' * 4
        return '{}{}'.format(indent, err[1])

    def report_error(self, test, err):
        self.stream.write('ERROR %s\n%s\n' % (self._testTimeString(test), self._error_summary(err)))

    def report_failure(self, test, err):
        self.stream.write(' FAIL %s\n%s\n' % (self._testTimeString(test), self._error_summary(err)))

    def report_known_failure(self, test, err):
        self.stream.write('XFAIL %s\n%s\n' % (self._testTimeString(test), self._error_summary(err)))

    def report_unexpected_success(self, test, reason):
        self.stream.write(' FAIL %s\n%s: %s\n' % (self._testTimeString(test), 'Unexpected success. Should have failed', reason))

    def report_success(self, test):
        self.stream.write('   OK %s\n' % self._testTimeString(test))
        for bench_called, stats in getattr(test, '_benchcalls', []):
            self.stream.write('LSProf output for %s(%s, %s)\n' % bench_called)
            stats.pprint(file=self.stream)
        self.stream.flush()

    def report_skip(self, test, reason):
        self.stream.write(' SKIP %s\n%s\n' % (self._testTimeString(test), reason))

    def report_not_applicable(self, test, reason):
        self.stream.write('  N/A %s\n    %s\n' % (self._testTimeString(test), reason))

    def report_unsupported(self, test, feature):
        """test cannot be run because feature is missing."""
        self.stream.write("NODEP %s\n    The feature '%s' is not available.\n" % (self._testTimeString(test), feature))