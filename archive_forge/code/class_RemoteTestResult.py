import argparse
import ctypes
import faulthandler
import hashlib
import io
import itertools
import logging
import multiprocessing
import os
import pickle
import random
import sys
import textwrap
import unittest
from collections import defaultdict
from contextlib import contextmanager
from importlib import import_module
from io import StringIO
import sqlparse
import django
from django.core.management import call_command
from django.db import connections
from django.test import SimpleTestCase, TestCase
from django.test.utils import NullTimeKeeper, TimeKeeper, iter_test_cases
from django.test.utils import setup_databases as _setup_databases
from django.test.utils import setup_test_environment
from django.test.utils import teardown_databases as _teardown_databases
from django.test.utils import teardown_test_environment
from django.utils.datastructures import OrderedSet
from django.utils.version import PY312
class RemoteTestResult(unittest.TestResult):
    """
    Extend unittest.TestResult to record events in the child processes so they
    can be replayed in the parent process. Events include things like which
    tests succeeded or failed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dummy_list = DummyList()
        self.failures = dummy_list
        self.errors = dummy_list
        self.skipped = dummy_list
        self.expectedFailures = dummy_list
        self.unexpectedSuccesses = dummy_list
        if tblib is not None:
            tblib.pickling_support.install()
        self.events = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_stdout_buffer', None)
        state.pop('_stderr_buffer', None)
        state.pop('_original_stdout', None)
        state.pop('_original_stderr', None)
        return state

    @property
    def test_index(self):
        return self.testsRun - 1

    def _confirm_picklable(self, obj):
        """
        Confirm that obj can be pickled and unpickled as multiprocessing will
        need to pickle the exception in the child process and unpickle it in
        the parent process. Let the exception rise, if not.
        """
        pickle.loads(pickle.dumps(obj))

    def _print_unpicklable_subtest(self, test, subtest, pickle_exc):
        print('\nSubtest failed:\n\n    test: {}\n subtest: {}\n\nUnfortunately, the subtest that failed cannot be pickled, so the parallel\ntest runner cannot handle it cleanly. Here is the pickling error:\n\n> {}\n\nYou should re-run this test with --parallel=1 to reproduce the failure\nwith a cleaner failure message.\n'.format(test, subtest, pickle_exc))

    def check_picklable(self, test, err):
        try:
            self._confirm_picklable(err)
        except Exception as exc:
            original_exc_txt = repr(err[1])
            original_exc_txt = textwrap.fill(original_exc_txt, 75, initial_indent='    ', subsequent_indent='    ')
            pickle_exc_txt = repr(exc)
            pickle_exc_txt = textwrap.fill(pickle_exc_txt, 75, initial_indent='    ', subsequent_indent='    ')
            if tblib is None:
                print('\n\n{} failed:\n\n{}\n\nUnfortunately, tracebacks cannot be pickled, making it impossible for the\nparallel test runner to handle this exception cleanly.\n\nIn order to see the traceback, you should install tblib:\n\n    python -m pip install tblib\n'.format(test, original_exc_txt))
            else:
                print("\n\n{} failed:\n\n{}\n\nUnfortunately, the exception it raised cannot be pickled, making it impossible\nfor the parallel test runner to handle it cleanly.\n\nHere's the error encountered while trying to pickle the exception:\n\n{}\n\nYou should re-run this test with the --parallel=1 option to reproduce the\nfailure and get a correct traceback.\n".format(test, original_exc_txt, pickle_exc_txt))
            raise

    def check_subtest_picklable(self, test, subtest):
        try:
            self._confirm_picklable(subtest)
        except Exception as exc:
            self._print_unpicklable_subtest(test, subtest, exc)
            raise

    def startTestRun(self):
        super().startTestRun()
        self.events.append(('startTestRun',))

    def stopTestRun(self):
        super().stopTestRun()
        self.events.append(('stopTestRun',))

    def startTest(self, test):
        super().startTest(test)
        self.events.append(('startTest', self.test_index))

    def stopTest(self, test):
        super().stopTest(test)
        self.events.append(('stopTest', self.test_index))

    def addDuration(self, test, elapsed):
        super().addDuration(test, elapsed)
        self.events.append(('addDuration', self.test_index, elapsed))

    def addError(self, test, err):
        self.check_picklable(test, err)
        self.events.append(('addError', self.test_index, err))
        super().addError(test, err)

    def addFailure(self, test, err):
        self.check_picklable(test, err)
        self.events.append(('addFailure', self.test_index, err))
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.check_picklable(test, err)
            self.check_subtest_picklable(test, subtest)
            self.events.append(('addSubTest', self.test_index, subtest, err))
        super().addSubTest(test, subtest, err)

    def addSuccess(self, test):
        self.events.append(('addSuccess', self.test_index))
        super().addSuccess(test)

    def addSkip(self, test, reason):
        self.events.append(('addSkip', self.test_index, reason))
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        if tblib is None:
            err = (err[0], err[1], None)
        self.check_picklable(test, err)
        self.events.append(('addExpectedFailure', self.test_index, err))
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.events.append(('addUnexpectedSuccess', self.test_index))
        super().addUnexpectedSuccess(test)

    def wasSuccessful(self):
        """Tells whether or not this result was a success."""
        failure_types = {'addError', 'addFailure', 'addSubTest', 'addUnexpectedSuccess'}
        return all((e[0] not in failure_types for e in self.events))

    def _exc_info_to_string(self, err, test):
        return ''