from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WarnOnMultipleStringArgsToOneOfTest(ParseTestCase):
    """
     - warn_on_multiple_string_args_to_oneof - flag to enable warnings whan oneOf is
       incorrectly called with multiple str arguments (default=True)
    """

    def runTest(self):
        import pyparsing as pp
        pp.__diag__.warn_on_multiple_string_args_to_oneof = True
        if PY_3:
            with self.assertWarns(UserWarning, msg='failed to warn when incorrectly calling oneOf(string, string)'):
                a = pp.oneOf('A', 'B')