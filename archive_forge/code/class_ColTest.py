from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ColTest(ParseTestCase):

    def runTest(self):
        test = '*\n* \n*   ALF\n*\n'
        initials = [c for i, c in enumerate(test) if pp.col(i, test) == 1]
        print_(initials)
        self.assertTrue(len(initials) == 4 and all((c == '*' for c in initials)), 'fail col test')