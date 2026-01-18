from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
class TreeReporter(Reporter):
    """
    Print out the tests in the form a tree.

    Tests are indented according to which class and module they belong.
    Results are printed in ANSI color.
    """
    currentLine = ''
    indent = '  '
    columns = 79
    FAILURE = 'red'
    ERROR = 'red'
    TODO = 'blue'
    SKIP = 'blue'
    TODONE = 'red'
    SUCCESS = 'green'

    def __init__(self, stream=sys.stdout, *args, **kwargs):
        super().__init__(stream, *args, **kwargs)
        self._lastTest = []
        for colorizer in [_Win32Colorizer, _AnsiColorizer, _NullColorizer]:
            if colorizer.supported(stream):
                self._colorizer = colorizer(stream)
                break

    def getDescription(self, test):
        """
        Return the name of the method which 'test' represents.  This is
        what gets displayed in the leaves of the tree.

        e.g. getDescription(TestCase('test_foo')) ==> test_foo
        """
        return test.id().split('.')[-1]

    def addSuccess(self, test):
        super().addSuccess(test)
        self.endLine('[OK]', self.SUCCESS)

    def addError(self, *args):
        super().addError(*args)
        self.endLine('[ERROR]', self.ERROR)

    def addFailure(self, *args):
        super().addFailure(*args)
        self.endLine('[FAIL]', self.FAILURE)

    def addSkip(self, *args):
        super().addSkip(*args)
        self.endLine('[SKIPPED]', self.SKIP)

    def addExpectedFailure(self, *args):
        super().addExpectedFailure(*args)
        self.endLine('[TODO]', self.TODO)

    def addUnexpectedSuccess(self, *args):
        super().addUnexpectedSuccess(*args)
        self.endLine('[SUCCESS!?!]', self.TODONE)

    def _write(self, format, *args):
        if args:
            format = format % args
        self.currentLine = format
        super()._write(self.currentLine)

    def _getPreludeSegments(self, testID):
        """
        Return a list of all non-leaf segments to display in the tree.

        Normally this is the module and class name.
        """
        segments = testID.split('.')[:-1]
        if len(segments) == 0:
            return segments
        segments = [seg for seg in ('.'.join(segments[:-1]), segments[-1]) if len(seg) > 0]
        return segments

    def _testPrelude(self, testID):
        """
        Write the name of the test to the stream, indenting it appropriately.

        If the test is the first test in a new 'branch' of the tree, also
        write all of the parents in that branch.
        """
        segments = self._getPreludeSegments(testID)
        indentLevel = 0
        for seg in segments:
            if indentLevel < len(self._lastTest):
                if seg != self._lastTest[indentLevel]:
                    self._write(f'{self.indent * indentLevel}{seg}\n')
            else:
                self._write(f'{self.indent * indentLevel}{seg}\n')
            indentLevel += 1
        self._lastTest = segments

    def cleanupErrors(self, errs):
        self._colorizer.write('    cleanup errors', self.ERROR)
        self.endLine('[ERROR]', self.ERROR)
        super().cleanupErrors(errs)

    def upDownError(self, method, error, warn, printStatus):
        self._colorizer.write('  %s' % method, self.ERROR)
        if printStatus:
            self.endLine('[ERROR]', self.ERROR)
        super().upDownError(method, error, warn, printStatus)

    def startTest(self, test):
        """
        Called when C{test} starts. Writes the tests name to the stream using
        a tree format.
        """
        self._testPrelude(test.id())
        self._write('%s%s ... ' % (self.indent * len(self._lastTest), self.getDescription(test)))
        super().startTest(test)

    def endLine(self, message, color):
        """
        Print 'message' in the given color.

        @param message: A string message, usually '[OK]' or something similar.
        @param color: A string color, 'red', 'green' and so forth.
        """
        spaces = ' ' * (self.columns - len(self.currentLine) - len(message))
        super()._write(spaces)
        self._colorizer.write(message, color)
        super()._write('\n')

    def _printSummary(self):
        """
        Print a line summarising the test results to the stream, and color the
        status result.
        """
        summary = self._getSummary()
        if self.wasSuccessful():
            status = 'PASSED'
            color = self.SUCCESS
        else:
            status = 'FAILED'
            color = self.FAILURE
        self._colorizer.write(status, color)
        self._write('%s\n', summary)