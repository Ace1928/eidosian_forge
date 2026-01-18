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
class DebugRunner(DocTestRunner):
    """Run doc tests but raise an exception as soon as there is a failure.

       If an unexpected exception occurs, an UnexpectedException is raised.
       It contains the test, the example, and the original exception:

         >>> runner = DebugRunner(verbose=False)
         >>> test = DocTestParser().get_doctest('>>> raise KeyError\\n42',
         ...                                    {}, 'foo', 'foo.py', 0)
         >>> try:
         ...     runner.run(test)
         ... except UnexpectedException as f:
         ...     failure = f

         >>> failure.test is test
         True

         >>> failure.example.want
         '42\\n'

         >>> exc_info = failure.exc_info
         >>> raise exc_info[1] # Already has the traceback
         Traceback (most recent call last):
         ...
         KeyError

       We wrap the original exception to give the calling application
       access to the test and example information.

       If the output doesn't match, then a DocTestFailure is raised:

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 1
         ...      >>> x
         ...      2
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> try:
         ...    runner.run(test)
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

       If a failure or error occurs, the globals are left intact:

         >>> del test.globs['__builtins__']
         >>> test.globs
         {'x': 1}

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 2
         ...      >>> raise KeyError
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> runner.run(test)
         Traceback (most recent call last):
         ...
         doctest.UnexpectedException: <DocTest foo from foo.py:0 (2 examples)>

         >>> del test.globs['__builtins__']
         >>> test.globs
         {'x': 2}

       But the globals are cleared if there is no error:

         >>> test = DocTestParser().get_doctest('''
         ...      >>> x = 2
         ...      ''', {}, 'foo', 'foo.py', 0)

         >>> runner.run(test)
         TestResults(failed=0, attempted=1)

         >>> test.globs
         {}

       """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        r = DocTestRunner.run(self, test, compileflags, out, False)
        if clear_globs:
            test.globs.clear()
        return r

    def report_unexpected_exception(self, out, test, example, exc_info):
        raise UnexpectedException(test, example, exc_info)

    def report_failure(self, out, test, example, got):
        raise DocTestFailure(test, example, got)