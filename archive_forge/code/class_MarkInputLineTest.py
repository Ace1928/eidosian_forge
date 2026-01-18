from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class MarkInputLineTest(ParseTestCase):

    def runTest(self):
        samplestr1 = 'DOB 100-10-2010;more garbage\nID PARI12345678;more garbage'
        from pyparsing import Regex
        dob_ref = 'DOB' + Regex('\\d{2}-\\d{2}-\\d{4}')('dob')
        try:
            res = dob_ref.parseString(samplestr1)
        except ParseException as pe:
            outstr = pe.markInputline()
            print_(outstr)
            self.assertEqual(outstr, 'DOB >!<100-10-2010;more garbage', 'did not properly create marked input line')
        else:
            self.assertEqual(False, 'test construction failed - should have raised an exception')