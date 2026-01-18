import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
class DocTestCase(unittest.TestCase):

    def __init__(self, test, optionflags=0, setUp=None, tearDown=None, checker=None):
        unittest.TestCase.__init__(self)
        self._dt_optionflags = optionflags
        self._dt_checker = checker
        self._dt_globs = test.globs.copy()
        self._dt_test = test
        self._dt_setUp = setUp
        self._dt_tearDown = tearDown

    def setUp(self):
        test = self._dt_test
        if self._dt_setUp is not None:
            self._dt_setUp(test)

    def tearDown(self):
        test = self._dt_test
        if self._dt_tearDown is not None:
            self._dt_tearDown(test)
        test.globs.clear()
        test.globs.update(self._dt_globs)

    def runTest(self):
        test = self._dt_test
        old = sys.stdout
        new = StringIO()
        optionflags = self._dt_optionflags
        if not optionflags & REPORTING_FLAGS:
            optionflags |= _unittest_reportflags
        runner = DocTestRunner(optionflags=optionflags, checker=self._dt_checker, verbose=False)
        try:
            runner.DIVIDER = '-' * 70
            failures, tries = runner.run(test, out=new.write, clear_globs=False)
        finally:
            sys.stdout = old
        if failures:
            raise self.failureException(self.format_failure(new.getvalue()))

    def format_failure(self, err):
        test = self._dt_test
        if test.lineno is None:
            lineno = 'unknown line number'
        else:
            lineno = '%s' % test.lineno
        lname = '.'.join(test.name.split('.')[-1:])
        return 'Failed doctest test for %s\n  File "%s", line %s, in %s\n\n%s' % (test.name, test.filename, lineno, lname, err)

    def debug(self):
        """Run the test case without results and without catching exceptions

           The unit test framework includes a debug method on test cases
           and test suites to support post-mortem debugging.  The test code
           is run in such a way that errors are not caught.  This way a
           caller can catch the errors and initiate post-mortem debugging.

           The DocTestCase provides a debug method that raises
           UnexpectedException errors if there is an unexpected
           exception:

             >>> test = DocTestParser().get_doctest('>>> raise KeyError\\n42',
             ...                {}, 'foo', 'foo.py', 0)
             >>> case = DocTestCase(test)
             >>> try:
             ...     case.debug()
             ... except UnexpectedException as f:
             ...     failure = f

           The UnexpectedException contains the test, the example, and
           the original exception:

             >>> failure.test is test
             True

             >>> failure.example.want
             '42\\n'

             >>> exc_info = failure.exc_info
             >>> raise exc_info[1] # Already has the traceback
             Traceback (most recent call last):
             ...
             KeyError

           If the output doesn't match, then a DocTestFailure is raised:

             >>> test = DocTestParser().get_doctest('''
             ...      >>> x = 1
             ...      >>> x
             ...      2
             ...      ''', {}, 'foo', 'foo.py', 0)
             >>> case = DocTestCase(test)

             >>> try:
             ...    case.debug()
             ... except DocTestFailure as f:
             ...    failure = f

           DocTestFailure objects provide access to the test:

             >>> failure.test is test
             True

           As well as to the example:

             >>> failure.example.want
             '2\\n'

           and the actual output:

             >>> failure.got
             '1\\n'

           """
        self.setUp()
        runner = DebugRunner(optionflags=self._dt_optionflags, checker=self._dt_checker, verbose=False)
        runner.run(self._dt_test, clear_globs=False)
        self.tearDown()

    def id(self):
        return self._dt_test.name

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self._dt_test == other._dt_test and self._dt_optionflags == other._dt_optionflags and (self._dt_setUp == other._dt_setUp) and (self._dt_tearDown == other._dt_tearDown) and (self._dt_checker == other._dt_checker)

    def __hash__(self):
        return hash((self._dt_optionflags, self._dt_setUp, self._dt_tearDown, self._dt_checker))

    def __repr__(self):
        name = self._dt_test.name.split('.')
        return '%s (%s)' % (name[-1], '.'.join(name[:-1]))
    __str__ = object.__str__

    def shortDescription(self):
        return 'Doctest: ' + self._dt_test.name