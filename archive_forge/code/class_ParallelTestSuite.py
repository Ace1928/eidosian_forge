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
class ParallelTestSuite(unittest.TestSuite):
    """
    Run a series of tests in parallel in several processes.

    While the unittest module's documentation implies that orchestrating the
    execution of tests is the responsibility of the test runner, in practice,
    it appears that TestRunner classes are more concerned with formatting and
    displaying test results.

    Since there are fewer use cases for customizing TestSuite than TestRunner,
    implementing parallelization at the level of the TestSuite improves
    interoperability with existing custom test runners. A single instance of a
    test runner can still collect results from all tests without being aware
    that they have been run in parallel.
    """
    init_worker = _init_worker
    process_setup = _process_setup_stub
    process_setup_args = ()
    run_subsuite = _run_subsuite
    runner_class = RemoteTestRunner

    def __init__(self, subsuites, processes, failfast=False, debug_mode=False, buffer=False):
        self.subsuites = subsuites
        self.processes = processes
        self.failfast = failfast
        self.debug_mode = debug_mode
        self.buffer = buffer
        self.initial_settings = None
        self.serialized_contents = None
        self.used_aliases = None
        super().__init__()

    def run(self, result):
        """
        Distribute TestCases across workers.

        Return an identifier of each TestCase with its result in order to use
        imap_unordered to show results as soon as they're available.

        To minimize pickling errors when getting results from workers:

        - pass back numeric indexes in self.subsuites instead of tests
        - make tracebacks picklable with tblib, if available

        Even with tblib, errors may still occur for dynamically created
        exception classes which cannot be unpickled.
        """
        self.initialize_suite()
        counter = multiprocessing.Value(ctypes.c_int, 0)
        pool = multiprocessing.Pool(processes=self.processes, initializer=self.init_worker.__func__, initargs=[counter, self.initial_settings, self.serialized_contents, self.process_setup.__func__, self.process_setup_args, self.debug_mode, self.used_aliases])
        args = [(self.runner_class, index, subsuite, self.failfast, self.buffer) for index, subsuite in enumerate(self.subsuites)]
        test_results = pool.imap_unordered(self.run_subsuite.__func__, args)
        while True:
            if result.shouldStop:
                pool.terminate()
                break
            try:
                subsuite_index, events = test_results.next(timeout=0.1)
            except multiprocessing.TimeoutError:
                continue
            except StopIteration:
                pool.close()
                break
            tests = list(self.subsuites[subsuite_index])
            for event in events:
                event_name = event[0]
                handler = getattr(result, event_name, None)
                if handler is None:
                    continue
                test = tests[event[1]]
                args = event[2:]
                handler(test, *args)
        pool.join()
        return result

    def __iter__(self):
        return iter(self.subsuites)

    def initialize_suite(self):
        if multiprocessing.get_start_method() == 'spawn':
            self.initial_settings = {alias: connections[alias].settings_dict for alias in connections}
            self.serialized_contents = {alias: connections[alias]._test_serialized_contents for alias in connections if alias in self.serialized_aliases}