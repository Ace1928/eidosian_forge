from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CaselessKeywordVsKeywordCaselessTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        frule = pp.Keyword('t', caseless=True) + pp.Keyword('yes', caseless=True)
        crule = pp.CaselessKeyword('t') + pp.CaselessKeyword('yes')
        flist = frule.searchString('not yes').asList()
        print_(flist)
        clist = crule.searchString('not yes').asList()
        print_(clist)
        self.assertEqual(flist, clist, 'CaselessKeyword not working the same as Keyword(caseless=True)')