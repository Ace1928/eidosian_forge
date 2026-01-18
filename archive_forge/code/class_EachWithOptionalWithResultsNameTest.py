from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class EachWithOptionalWithResultsNameTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Optional
        result = (Optional('foo')('one') & Optional('bar')('two')).parseString('bar foo')
        print_(result.dump())
        self.assertEqual(sorted(result.keys()), ['one', 'two'])