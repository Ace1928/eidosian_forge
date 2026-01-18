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
class ExtendedTestResult(testtools.TextTestResult):
    """Accepts, reports and accumulates the results of running tests.

    Compared to the unittest version this class adds support for
    profiling, benchmarking, stopping as soon as a test fails,  and
    skipping tests.  There are further-specialized subclasses for
    different types of display.

    When a test finishes, in whatever way, it calls one of the addSuccess,
    addFailure or addError methods.  These in turn may redirect to a more
    specific case for the special test results supported by our extended
    tests.

    Note that just one of these objects is fed the results from many tests.
    """
    stop_early = False

    def __init__(self, stream, descriptions, verbosity, bench_history=None, strict=False):
        """Construct new TestResult.

        :param bench_history: Optionally, a writable file object to accumulate
            benchmark results.
        """
        testtools.TextTestResult.__init__(self, stream)
        if bench_history is not None:
            from breezy.version import _get_bzr_source_tree
            src_tree = _get_bzr_source_tree()
            if src_tree:
                try:
                    revision_id = src_tree.get_parent_ids()[0]
                except IndexError:
                    revision_id = b''
            else:
                revision_id = b''
            bench_history.write('--date {} {}\n'.format(time.time(), revision_id))
        self._bench_history = bench_history
        self.ui = ui.ui_factory
        self.num_tests = 0
        self.error_count = 0
        self.failure_count = 0
        self.known_failure_count = 0
        self.skip_count = 0
        self.not_applicable_count = 0
        self.unsupported = {}
        self.count = 0
        self._overall_start_time = time.time()
        self._strict = strict
        self._first_thread_leaker_id = None
        self._tests_leaking_threads_count = 0
        self._traceback_from_test = None

    def stopTestRun(self):
        run = self.testsRun
        actionTaken = 'Ran'
        stopTime = time.time()
        timeTaken = stopTime - self.startTime
        self._show_list('ERROR', self.errors)
        self._show_list('FAIL', self.failures)
        self.stream.write(self.sep2)
        self.stream.write('%s %d test%s in %.3fs\n\n' % (actionTaken, run, run != 1 and 's' or '', timeTaken))
        if not self.wasSuccessful():
            self.stream.write('FAILED (')
            failed, errored = map(len, (self.failures, self.errors))
            if failed:
                self.stream.write('failures=%d' % failed)
            if errored:
                if failed:
                    self.stream.write(', ')
                self.stream.write('errors=%d' % errored)
            if self.known_failure_count:
                if failed or errored:
                    self.stream.write(', ')
                self.stream.write('known_failure_count=%d' % self.known_failure_count)
            self.stream.write(')\n')
        elif self.known_failure_count:
            self.stream.write('OK (known_failures=%d)\n' % self.known_failure_count)
        else:
            self.stream.write('OK\n')
        if self.skip_count > 0:
            skipped = self.skip_count
            self.stream.write('%d test%s skipped\n' % (skipped, skipped != 1 and 's' or ''))
        if self.unsupported:
            for feature, count in sorted(self.unsupported.items()):
                self.stream.write("Missing feature '%s' skipped %d tests.\n" % (feature, count))
        if self._strict:
            ok = self.wasStrictlySuccessful()
        else:
            ok = self.wasSuccessful()
        if self._first_thread_leaker_id:
            self.stream.write('%s is leaking threads among %d leaking tests.\n' % (self._first_thread_leaker_id, self._tests_leaking_threads_count))
            self.stream.write('%d non-main threads were left active in the end.\n' % (len(self._active_threads) - 1))

    def getDescription(self, test):
        return test.id()

    def _extractBenchmarkTime(self, testCase, details=None):
        """Add a benchmark time for the current test case."""
        if details and 'benchtime' in details:
            return float(''.join(details['benchtime'].iter_bytes()))
        return getattr(testCase, '_benchtime', None)

    def _delta_to_float(self, a_timedelta, precision):
        shift = 10 ** precision
        return math.ceil((a_timedelta.days * 86400.0 + a_timedelta.seconds + a_timedelta.microseconds / 1000000.0) * shift) / shift

    def _elapsedTestTimeString(self):
        """Return time string for overall time the current test has taken."""
        return self._formatTime(self._delta_to_float(self._now() - self._start_datetime, 3))

    def _testTimeString(self, testCase):
        benchmark_time = self._extractBenchmarkTime(testCase)
        if benchmark_time is not None:
            return self._formatTime(benchmark_time) + '*'
        else:
            return self._elapsedTestTimeString()

    def _formatTime(self, seconds):
        """Format seconds as milliseconds with leading spaces."""
        return '%8dms' % (1000 * seconds)

    def _shortened_test_description(self, test):
        what = test.id()
        what = re.sub('^breezy\\.tests\\.', '', what)
        return what

    def _record_traceback_from_test(self, exc_info):
        """Store the traceback from passed exc_info tuple till"""
        self._traceback_from_test = exc_info[2]

    def startTest(self, test):
        super().startTest(test)
        if self.count == 0:
            self.startTests()
        self.count += 1
        self.report_test_start(test)
        test.number = self.count
        self._recordTestStartTime()
        addOnException = getattr(test, 'addOnException', None)
        if addOnException is not None:
            addOnException(self._record_traceback_from_test)
        if isinstance(test, TestCase):
            test.addCleanup(self._check_leaked_threads, test)

    def stopTest(self, test):
        super().stopTest(test)
        getDetails = getattr(test, 'getDetails', None)
        if getDetails is not None:
            getDetails().clear()
        _clear__type_equality_funcs(test)
        self._traceback_from_test = None

    def startTests(self):
        self.report_tests_starting()
        self._active_threads = threading.enumerate()

    def _check_leaked_threads(self, test):
        """See if any threads have leaked since last call

        A sample of live threads is stored in the _active_threads attribute,
        when this method runs it compares the current live threads and any not
        in the previous sample are treated as having leaked.
        """
        now_active_threads = set(threading.enumerate())
        threads_leaked = now_active_threads.difference(self._active_threads)
        if threads_leaked:
            self._report_thread_leak(test, threads_leaked, now_active_threads)
            self._tests_leaking_threads_count += 1
            if self._first_thread_leaker_id is None:
                self._first_thread_leaker_id = test.id()
            self._active_threads = now_active_threads

    def _recordTestStartTime(self):
        """Record that a test has started."""
        self._start_datetime = self._now()

    def addError(self, test, err):
        """Tell result that test finished with an error.

        Called from the TestCase run() method when the test
        fails with an unexpected error.
        """
        self._post_mortem(self._traceback_from_test or err[2])
        super().addError(test, err)
        self.error_count += 1
        self.report_error(test, err)
        if self.stop_early:
            self.stop()

    def addFailure(self, test, err):
        """Tell result that test failed.

        Called from the TestCase run() method when the test
        fails because e.g. an assert() method failed.
        """
        self._post_mortem(self._traceback_from_test or err[2])
        super().addFailure(test, err)
        self.failure_count += 1
        self.report_failure(test, err)
        if self.stop_early:
            self.stop()

    def addSuccess(self, test, details=None):
        """Tell result that test completed successfully.

        Called from the TestCase run()
        """
        if self._bench_history is not None:
            benchmark_time = self._extractBenchmarkTime(test, details)
            if benchmark_time is not None:
                self._bench_history.write('{} {}\n'.format(self._formatTime(benchmark_time), test.id()))
        self.report_success(test)
        super().addSuccess(test)
        test._log_contents = ''

    def addExpectedFailure(self, test, err):
        self.known_failure_count += 1
        self.report_known_failure(test, err)

    def addUnexpectedSuccess(self, test, details=None):
        """Tell result the test unexpectedly passed, counting as a failure

        When the minimum version of testtools required becomes 0.9.8 this
        can be updated to use the new handling there.
        """
        super().addFailure(test, details=details)
        self.failure_count += 1
        self.report_unexpected_success(test, ''.join(details['reason'].iter_text()))
        if self.stop_early:
            self.stop()

    def addNotSupported(self, test, feature):
        """The test will not be run because of a missing feature.
        """
        self.unsupported.setdefault(str(feature), 0)
        self.unsupported[str(feature)] += 1
        self.report_unsupported(test, feature)

    def addSkip(self, test, reason):
        """A test has not run for 'reason'."""
        self.skip_count += 1
        self.report_skip(test, reason)

    def addNotApplicable(self, test, reason):
        self.not_applicable_count += 1
        self.report_not_applicable(test, reason)

    def _count_stored_tests(self):
        """Count of tests instances kept alive due to not succeeding"""
        return self.error_count + self.failure_count + self.known_failure_count

    def _post_mortem(self, tb=None):
        """Start a PDB post mortem session."""
        if os.environ.get('BRZ_TEST_PDB', None):
            import pdb
            pdb.post_mortem(tb)

    def progress(self, offset, whence):
        """The test is adjusting the count of tests to run."""
        if whence == SUBUNIT_SEEK_SET:
            self.num_tests = offset
        elif whence == SUBUNIT_SEEK_CUR:
            self.num_tests += offset
        else:
            raise errors.BzrError('Unknown whence %r' % whence)

    def report_tests_starting(self):
        """Display information before the test run begins"""
        bzr_path = osutils.realpath(sys.argv[0])
        self.stream.write('brz selftest: {}\n'.format(bzr_path))
        self.stream.write('   {}\n'.format(breezy.__path__[0]))
        self.stream.write('   bzr-{} python-{} {}\n'.format(breezy.version_string, breezy._format_version_tuple(sys.version_info), platform.platform(aliased=1)))
        self.stream.write('\n')

    def report_test_start(self, test):
        """Display information on the test just about to be run"""

    def _report_thread_leak(self, test, leaked_threads, active_threads):
        """Display information on a test that leaked one or more threads"""
        if 'threads' in selftest_debug_flags:
            self.stream.write('%s is leaking, active is now %d\n' % (test.id(), len(active_threads)))

    def startTestRun(self):
        self.startTime = time.time()

    def report_success(self, test):
        pass

    def wasStrictlySuccessful(self):
        if self.unsupported or self.known_failure_count:
            return False
        return self.wasSuccessful()