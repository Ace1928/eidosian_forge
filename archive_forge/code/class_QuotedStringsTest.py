from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class QuotedStringsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import sglQuotedString, dblQuotedString, quotedString, QuotedString
        testData = '\n                \'a valid single quoted string\'\n                \'an invalid single quoted string\n                 because it spans lines\'\n                "a valid double quoted string"\n                "an invalid double quoted string\n                 because it spans lines"\n            '
        print_(testData)
        sglStrings = [(t[0], b, e) for t, b, e in sglQuotedString.scanString(testData)]
        print_(sglStrings)
        self.assertTrue(len(sglStrings) == 1 and (sglStrings[0][1] == 17 and sglStrings[0][2] == 47), 'single quoted string failure')
        dblStrings = [(t[0], b, e) for t, b, e in dblQuotedString.scanString(testData)]
        print_(dblStrings)
        self.assertTrue(len(dblStrings) == 1 and (dblStrings[0][1] == 154 and dblStrings[0][2] == 184), 'double quoted string failure')
        allStrings = [(t[0], b, e) for t, b, e in quotedString.scanString(testData)]
        print_(allStrings)
        self.assertTrue(len(allStrings) == 2 and (allStrings[0][1] == 17 and allStrings[0][2] == 47) and (allStrings[1][1] == 154 and allStrings[1][2] == 184), 'quoted string failure')
        escapedQuoteTest = '\n                \'This string has an escaped (\\\') quote character\'\n                "This string has an escaped (\\") quote character"\n            '
        sglStrings = [(t[0], b, e) for t, b, e in sglQuotedString.scanString(escapedQuoteTest)]
        print_(sglStrings)
        self.assertTrue(len(sglStrings) == 1 and (sglStrings[0][1] == 17 and sglStrings[0][2] == 66), 'single quoted string escaped quote failure (%s)' % str(sglStrings[0]))
        dblStrings = [(t[0], b, e) for t, b, e in dblQuotedString.scanString(escapedQuoteTest)]
        print_(dblStrings)
        self.assertTrue(len(dblStrings) == 1 and (dblStrings[0][1] == 83 and dblStrings[0][2] == 132), 'double quoted string escaped quote failure (%s)' % str(dblStrings[0]))
        allStrings = [(t[0], b, e) for t, b, e in quotedString.scanString(escapedQuoteTest)]
        print_(allStrings)
        self.assertTrue(len(allStrings) == 2 and (allStrings[0][1] == 17 and allStrings[0][2] == 66 and (allStrings[1][1] == 83) and (allStrings[1][2] == 132)), 'quoted string escaped quote failure (%s)' % [str(s[0]) for s in allStrings])
        dblQuoteTest = '\n                \'This string has an doubled (\'\') quote character\'\n                "This string has an doubled ("") quote character"\n            '
        sglStrings = [(t[0], b, e) for t, b, e in sglQuotedString.scanString(dblQuoteTest)]
        print_(sglStrings)
        self.assertTrue(len(sglStrings) == 1 and (sglStrings[0][1] == 17 and sglStrings[0][2] == 66), 'single quoted string escaped quote failure (%s)' % str(sglStrings[0]))
        dblStrings = [(t[0], b, e) for t, b, e in dblQuotedString.scanString(dblQuoteTest)]
        print_(dblStrings)
        self.assertTrue(len(dblStrings) == 1 and (dblStrings[0][1] == 83 and dblStrings[0][2] == 132), 'double quoted string escaped quote failure (%s)' % str(dblStrings[0]))
        allStrings = [(t[0], b, e) for t, b, e in quotedString.scanString(dblQuoteTest)]
        print_(allStrings)
        self.assertTrue(len(allStrings) == 2 and (allStrings[0][1] == 17 and allStrings[0][2] == 66 and (allStrings[1][1] == 83) and (allStrings[1][2] == 132)), 'quoted string escaped quote failure (%s)' % [str(s[0]) for s in allStrings])
        print_('testing catastrophic RE backtracking in implementation of dblQuotedString')
        for expr, test_string in [(dblQuotedString, '"' + '\\xff' * 500), (sglQuotedString, "'" + '\\xff' * 500), (quotedString, '"' + '\\xff' * 500), (quotedString, "'" + '\\xff' * 500), (QuotedString('"'), '"' + '\\xff' * 500), (QuotedString("'"), "'" + '\\xff' * 500)]:
            expr.parseString(test_string + test_string[0])
            try:
                expr.parseString(test_string)
            except Exception:
                continue