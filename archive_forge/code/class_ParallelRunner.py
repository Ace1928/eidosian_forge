from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
class ParallelRunner(ColouredTextRunner):

    @staticmethod
    def _parallelize(suite):

        def fdopen(fd, mode, *kwds):
            stream = orig_fdopen(fd, mode)
            atexit.register(stream.close)
            return stream
        orig_fdopen = os.fdopen
        concurrencytest.os.fdopen = fdopen
        forker = concurrencytest.fork_for_tests(NWORKERS)
        return concurrencytest.ConcurrentTestSuite(suite, forker)

    @staticmethod
    def _split_suite(suite):
        serial = unittest.TestSuite()
        parallel = unittest.TestSuite()
        for test in suite:
            if test.countTestCases() == 0:
                continue
            if isinstance(test, unittest.TestSuite):
                test_class = test._tests[0].__class__
            elif isinstance(test, unittest.TestCase):
                test_class = test
            else:
                raise TypeError("can't recognize type %r" % test)
            if getattr(test_class, '_serialrun', False):
                serial.addTest(test)
            else:
                parallel.addTest(test)
        return (serial, parallel)

    def run(self, suite):
        ser_suite, par_suite = self._split_suite(suite)
        par_suite = self._parallelize(par_suite)
        cprint('starting parallel tests using %s workers' % NWORKERS, 'green', bold=True)
        t = time.time()
        par = self._run(par_suite)
        par_elapsed = time.time() - t
        orphans = psutil.Process().children()
        gone, alive = psutil.wait_procs(orphans, timeout=1)
        if alive:
            cprint('alive processes %s' % alive, 'red')
            reap_children()
        t = time.time()
        ser = self._run(ser_suite)
        ser_elapsed = time.time() - t
        if not par.wasSuccessful() and ser_suite.countTestCases() > 0:
            par.printErrors()
        par_fails, par_errs, par_skips = map(len, (par.failures, par.errors, par.skipped))
        ser_fails, ser_errs, ser_skips = map(len, (ser.failures, ser.errors, ser.skipped))
        print(textwrap.dedent('\n            +----------+----------+----------+----------+----------+----------+\n            |          |    total | failures |   errors |  skipped |     time |\n            +----------+----------+----------+----------+----------+----------+\n            | parallel |      %3s |      %3s |      %3s |      %3s |    %.2fs |\n            +----------+----------+----------+----------+----------+----------+\n            | serial   |      %3s |      %3s |      %3s |      %3s |    %.2fs |\n            +----------+----------+----------+----------+----------+----------+\n            ' % (par.testsRun, par_fails, par_errs, par_skips, par_elapsed, ser.testsRun, ser_fails, ser_errs, ser_skips, ser_elapsed)))
        print('Ran %s tests in %.3fs using %s workers' % (par.testsRun + ser.testsRun, par_elapsed + ser_elapsed, NWORKERS))
        ok = par.wasSuccessful() and ser.wasSuccessful()
        self._exit(ok)