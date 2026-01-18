import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
class NumbaTestProgram(unittest.main):
    """
    A TestProgram subclass adding the following options:
    * a -R option to enable reference leak detection
    * a --profile option to enable profiling of the test run
    * a -m option for parallel execution
    * a -l option to (only) list tests

    Currently the options are only added in 3.4+.
    """
    refleak = False
    profile = False
    multiprocess = False
    useslice = None
    list = False
    tags = None
    exclude_tags = None
    random_select = None
    random_seed = 42

    def __init__(self, *args, **kwargs):
        topleveldir = kwargs.pop('topleveldir', None)
        kwargs['testLoader'] = TestLoader(topleveldir)
        sys.warnoptions.append(':x')
        self.nomultiproc = kwargs.pop('nomultiproc', False)
        super(NumbaTestProgram, self).__init__(*args, **kwargs)

    def _getParentArgParser(self):
        parser = super(NumbaTestProgram, self)._getParentArgParser()
        if self.testRunner is None:
            parser.add_argument('-R', '--refleak', dest='refleak', action='store_true', help='Detect reference / memory leaks')
        parser.add_argument('-m', '--multiprocess', dest='multiprocess', nargs='?', type=int, const=multiprocessing.cpu_count(), help='Parallelize tests')
        parser.add_argument('-l', '--list', dest='list', action='store_true', help='List tests without running them')
        parser.add_argument('--tags', dest='tags', type=str, help='Comma-separated list of tags to select a subset of the test suite')
        parser.add_argument('--exclude-tags', dest='exclude_tags', type=str, help='Comma-separated list of tags to de-select a subset of the test suite')
        parser.add_argument('--random', dest='random_select', type=float, help='Random proportion of tests to select')
        parser.add_argument('--profile', dest='profile', action='store_true', help='Profile the test run')
        parser.add_argument('-j', '--slice', dest='useslice', nargs='?', type=str, const='None', help='Shard the test sequence')

        def git_diff_str(x):
            if x != 'ancestor':
                raise ValueError('invalid option for --gitdiff')
            return x
        parser.add_argument('-g', '--gitdiff', dest='gitdiff', type=git_diff_str, default=False, nargs='?', help='Run tests from changes made against origin/release0.59 as identified by `git diff`. If set to "ancestor", the diff compares against the common ancestor.')
        return parser

    def _handle_tags(self, argv, tagstr):
        found = None
        for x in argv:
            if tagstr in x:
                if found is None:
                    found = x
                else:
                    raise ValueError('argument %s supplied repeatedly' % tagstr)
        if found is not None:
            posn = argv.index(found)
            try:
                if found == tagstr:
                    tag_args = argv[posn + 1].strip()
                    argv.remove(tag_args)
                elif '=' in found:
                    tag_args = found.split('=')[1].strip()
                else:
                    raise AssertionError('unreachable')
            except IndexError:
                msg = '%s requires at least one tag to be specified'
                raise ValueError(msg % tagstr)
            if tag_args.startswith('-'):
                raise ValueError("tag starts with '-', probably a syntax error")
            if '=' in tag_args:
                msg = "%s argument contains '=', probably a syntax error"
                raise ValueError(msg % tagstr)
            attr = tagstr[2:].replace('-', '_')
            setattr(self, attr, tag_args)
            argv.remove(found)

    def parseArgs(self, argv):
        if '-l' in argv:
            argv.remove('-l')
            self.list = True
        super(NumbaTestProgram, self).parseArgs(argv)
        if not hasattr(self, 'test') or not self.test.countTestCases():
            self.testNames = (self.defaultTest,)
            self.createTests()
        if self.tags:
            tags = [s.strip() for s in self.tags.split(',')]
            self.test = _choose_tagged_tests(self.test, tags, mode='include')
        if self.exclude_tags:
            tags = [s.strip() for s in self.exclude_tags.split(',')]
            self.test = _choose_tagged_tests(self.test, tags, mode='exclude')
        if self.random_select:
            self.test = _choose_random_tests(self.test, self.random_select, self.random_seed)
        if self.gitdiff is not False:
            self.test = _choose_gitdiff_tests(self.test, use_common_ancestor=self.gitdiff == 'ancestor')
        if self.verbosity <= 0:
            self.buffer = True

    def _do_discovery(self, argv, Loader=None):
        return

    def runTests(self):
        if self.refleak:
            self.testRunner = RefleakTestRunner
            if not hasattr(sys, 'gettotalrefcount'):
                warnings.warn('detecting reference leaks requires a debug build of Python, only memory leaks will be detected')
        elif self.list:
            self.testRunner = TestLister(self.useslice)
        elif self.testRunner is None:
            self.testRunner = BasicTestRunner(self.useslice, verbosity=self.verbosity, failfast=self.failfast, buffer=self.buffer)
        if self.multiprocess and (not self.nomultiproc):
            if self.multiprocess < 1:
                msg = 'Value specified for the number of processes to use in running the suite must be > 0'
                raise ValueError(msg)
            self.testRunner = ParallelTestRunner(runner.TextTestRunner, self.multiprocess, self.useslice, verbosity=self.verbosity, failfast=self.failfast, buffer=self.buffer)

        def run_tests_real():
            super(NumbaTestProgram, self).runTests()
        if self.profile:
            filename = os.path.splitext(os.path.basename(sys.modules['__main__'].__file__))[0] + '.prof'
            p = cProfile.Profile(timer=time.perf_counter)
            p.enable()
            try:
                p.runcall(run_tests_real)
            finally:
                p.disable()
                print('Writing test profile data into %r' % (filename,))
                p.dump_stats(filename)
        else:
            run_tests_real()