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
def __run(self, test, compileflags, out):
    """
        Run the examples in `test`.  Write the outcome of each example
        with one of the `DocTestRunner.report_*` methods, using the
        writer function `out`.  `compileflags` is the set of compiler
        flags that should be used to execute examples.  Return a tuple
        `(f, t)`, where `t` is the number of examples tried, and `f`
        is the number of examples that failed.  The examples are run
        in the namespace `test.globs`.
        """
    failures = tries = 0
    original_optionflags = self.optionflags
    SUCCESS, FAILURE, BOOM = range(3)
    check = self._checker.check_output
    for examplenum, example in enumerate(test.examples):
        quiet = self.optionflags & REPORT_ONLY_FIRST_FAILURE and failures > 0
        self.optionflags = original_optionflags
        if example.options:
            for optionflag, val in example.options.items():
                if val:
                    self.optionflags |= optionflag
                else:
                    self.optionflags &= ~optionflag
        if self.optionflags & SKIP:
            continue
        tries += 1
        if not quiet:
            self.report_start(out, test, example)
        filename = '<doctest %s[%d]>' % (test.name, examplenum)
        try:
            exec(compile(example.source, filename, 'single', compileflags, True), test.globs)
            self.debugger.set_continue()
            exception = None
        except KeyboardInterrupt:
            raise
        except:
            exception = sys.exc_info()
            self.debugger.set_continue()
        got = self._fakeout.getvalue()
        self._fakeout.truncate(0)
        outcome = FAILURE
        if exception is None:
            if check(example.want, got, self.optionflags):
                outcome = SUCCESS
        else:
            formatted_ex = traceback.format_exception_only(*exception[:2])
            if issubclass(exception[0], SyntaxError):
                exception_line_prefixes = (f'{exception[0].__qualname__}:', f'{exception[0].__module__}.{exception[0].__qualname__}:')
                exc_msg_index = next((index for index, line in enumerate(formatted_ex) if line.startswith(exception_line_prefixes)))
                formatted_ex = formatted_ex[exc_msg_index:]
            exc_msg = ''.join(formatted_ex)
            if not quiet:
                got += _exception_traceback(exception)
            if example.exc_msg is None:
                outcome = BOOM
            elif check(example.exc_msg, exc_msg, self.optionflags):
                outcome = SUCCESS
            elif self.optionflags & IGNORE_EXCEPTION_DETAIL:
                if check(_strip_exception_details(example.exc_msg), _strip_exception_details(exc_msg), self.optionflags):
                    outcome = SUCCESS
        if outcome is SUCCESS:
            if not quiet:
                self.report_success(out, test, example, got)
        elif outcome is FAILURE:
            if not quiet:
                self.report_failure(out, test, example, got)
            failures += 1
        elif outcome is BOOM:
            if not quiet:
                self.report_unexpected_exception(out, test, example, exception)
            failures += 1
        else:
            assert False, ('unknown outcome', outcome)
        if failures and self.optionflags & FAIL_FAST:
            break
    self.optionflags = original_optionflags
    self.__record_outcome(test, failures, tries)
    return TestResults(failures, tries)