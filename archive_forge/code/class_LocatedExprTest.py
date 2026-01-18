from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class LocatedExprTest(ParseTestCase):

    def runTest(self):
        samplestr1 = 'DOB 10-10-2010;more garbage;ID PARI12345678  ;more garbage'
        from pyparsing import Word, alphanums, locatedExpr
        id_ref = locatedExpr('ID' + Word(alphanums, exact=12)('id'))
        res = id_ref.searchString(samplestr1)[0][0]
        print_(res.dump())
        self.assertEqual(samplestr1[res.locn_start:res.locn_end], 'ID PARI12345678', 'incorrect location calculation')