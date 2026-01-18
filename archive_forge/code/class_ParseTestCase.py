from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseTestCase(ppt.TestParseResultsAsserts, TestCase):

    def __init__(self):
        super(ParseTestCase, self).__init__(methodName='_runTest')
        self.expect_traceback = False
        self.expect_warning = False

    def _runTest(self):
        buffered_stdout = StringIO()
        try:
            with AutoReset(sys, 'stdout', 'stderr'):
                try:
                    if BUFFER_OUTPUT:
                        sys.stdout = buffered_stdout
                        sys.stderr = buffered_stdout
                    print_('>>>> Starting test', str(self))
                    with ppt.reset_pyparsing_context():
                        self.runTest()
                finally:
                    print_('<<<< End of test', str(self))
                    print_()
            output = buffered_stdout.getvalue()
            if 'Traceback' in output and (not self.expect_traceback):
                raise Exception('traceback in stdout')
            if 'Warning' in output and (not self.expect_warning):
                raise Exception('warning in stdout')
        except Exception as exc:
            if BUFFER_OUTPUT:
                print_()
                print_(buffered_stdout.getvalue())
            raise

    def runTest(self):
        pass

    def __str__(self):
        return self.__class__.__name__