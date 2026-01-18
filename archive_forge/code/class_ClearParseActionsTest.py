from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ClearParseActionsTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        ppc = pp.pyparsing_common
        realnum = ppc.real()
        self.assertEqual(realnum.parseString('3.14159')[0], 3.14159, 'failed basic real number parsing')
        realnum.setParseAction(None)
        self.assertEqual(realnum.parseString('3.14159')[0], '3.14159', 'failed clearing parse action')
        realnum.addParseAction(lambda t: '.' in t[0])
        self.assertEqual(realnum.parseString('3.14159')[0], True, 'failed setting new parse action after clearing parse action')