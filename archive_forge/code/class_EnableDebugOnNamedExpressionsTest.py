from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class EnableDebugOnNamedExpressionsTest(ParseTestCase):
    """
     - enable_debug_on_named_expressions - flag to auto-enable debug on all subsequent
       calls to ParserElement.setName() (default=False)
    """

    def runTest(self):
        import pyparsing as pp
        import textwrap
        test_stdout = StringIO()
        with AutoReset(sys, 'stdout', 'stderr'):
            sys.stdout = test_stdout
            sys.stderr = test_stdout
            pp.__diag__.enable_debug_on_named_expressions = True
            integer = pp.Word(pp.nums).setName('integer')
            integer[...].parseString('1 2 3')
        expected_debug_output = textwrap.dedent("            Match integer at loc 0(1,1)\n            Matched integer -> ['1']\n            Match integer at loc 1(1,2)\n            Matched integer -> ['2']\n            Match integer at loc 3(1,4)\n            Matched integer -> ['3']\n            Match integer at loc 5(1,6)\n            Exception raised:Expected integer, found end of text  (at char 5), (line:1, col:6)\n            ")
        output = test_stdout.getvalue()
        print_(output)
        self.assertEqual(output, expected_debug_output, 'failed to auto-enable debug on named expressions using enable_debug_on_named_expressions')