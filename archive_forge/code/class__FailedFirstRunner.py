import json
import re
import logging
class _FailedFirstRunner(object):
    """
    Test Runner to handle the failed-first (--failed-first) option.
    """
    cache_filename = '.runtests_lastfailed'

    def __init__(self, last_failed=False):
        self.last_failed = last_failed

    def main(self, argv, kwds):
        from numba.testing import run_tests
        prog = argv[0]
        argv = argv[1:]
        flags = [a for a in argv if a.startswith('-')]
        all_tests, failed_tests = self.find_last_failed(argv)
        if failed_tests:
            ft = 'There were {} previously failed tests'
            print(ft.format(len(failed_tests)))
            remaing_tests = [t for t in all_tests if t not in failed_tests]
            if self.last_failed:
                tests = list(failed_tests)
            else:
                tests = failed_tests + remaing_tests
        elif self.last_failed:
            tests = []
        else:
            tests = list(all_tests)
        if not tests:
            print('No tests to run')
            return True
        print('Running {} tests'.format(len(tests)))
        print('Flags', flags)
        result = run_tests([prog] + flags + tests, **kwds)
        if len(tests) == result.testsRun:
            self.save_failed_tests(result, all_tests)
        return result.wasSuccessful()

    def save_failed_tests(self, result, all_tests):
        print('Saving failed tests to {}'.format(self.cache_filename))
        cache = []
        failed = set()
        for case in result.errors + result.failures:
            failed.add(case[0].id())
        for t in all_tests:
            if t in failed:
                cache.append(t)
        with open(self.cache_filename, 'w') as fout:
            json.dump(cache, fout)

    def find_last_failed(self, argv):
        from numba.tests.support import captured_output
        listargv = ['-l'] + [a for a in argv if not a.startswith('-')]
        with captured_output('stdout') as stream:
            main(*listargv)
            pat = re.compile('^(\\w+\\.)+\\w+$')
            lines = stream.getvalue().splitlines()
        all_tests = [x for x in lines if pat.match(x) is not None]
        try:
            fobj = open(self.cache_filename)
        except OSError:
            failed_tests = []
        else:
            with fobj as fin:
                failed_tests = json.load(fin)
        return (all_tests, failed_tests)