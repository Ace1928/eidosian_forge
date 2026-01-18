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
class TextTestResult(ExtendedTestResult):
    """Displays progress and results of tests in text form"""

    def __init__(self, stream, descriptions, verbosity, bench_history=None, strict=None):
        ExtendedTestResult.__init__(self, stream, descriptions, verbosity, bench_history, strict)
        self.pb = self.ui.nested_progress_bar()
        self.pb.show_pct = False
        self.pb.show_spinner = False
        self.pb.show_eta = (False,)
        self.pb.show_count = False
        self.pb.show_bar = False
        self.pb.update_latency = 0
        self.pb.show_transport_activity = False

    def stopTestRun(self):
        self.pb.clear()
        self.pb.finished()
        super().stopTestRun()

    def report_tests_starting(self):
        super().report_tests_starting()
        self.pb.update('[test 0/%d] Starting' % self.num_tests)

    def _progress_prefix_text(self):
        a = '[%d' % self.count
        if self.num_tests:
            a += '/%d' % self.num_tests
        a += ' in '
        runtime = time.time() - self._overall_start_time
        if runtime >= 60:
            a += '%dm%ds' % (runtime / 60, runtime % 60)
        else:
            a += '%ds' % runtime
        total_fail_count = self.error_count + self.failure_count
        if total_fail_count:
            a += ', %d failed' % total_fail_count
        a += ']'
        return a

    def report_test_start(self, test):
        self.pb.update(self._progress_prefix_text() + ' ' + self._shortened_test_description(test))

    def _test_description(self, test):
        return self._shortened_test_description(test)

    def report_error(self, test, err):
        self.stream.write('ERROR: {}\n    {}\n'.format(self._test_description(test), err[1]))

    def report_failure(self, test, err):
        self.stream.write('FAIL: {}\n    {}\n'.format(self._test_description(test), err[1]))

    def report_known_failure(self, test, err):
        pass

    def report_unexpected_success(self, test, reason):
        self.stream.write('FAIL: {}\n    {}: {}\n'.format(self._test_description(test), 'Unexpected success. Should have failed', reason))

    def report_skip(self, test, reason):
        pass

    def report_not_applicable(self, test, reason):
        pass

    def report_unsupported(self, test, feature):
        """test cannot be run because feature is missing."""