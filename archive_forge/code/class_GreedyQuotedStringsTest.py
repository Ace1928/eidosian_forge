from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class GreedyQuotedStringsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import QuotedString, sglQuotedString, dblQuotedString, quotedString, delimitedList
        src = '           "string1", "strin""g2"\n           \'string1\', \'string2\'\n           ^string1^, ^string2^\n           <string1>, <string2>'
        testExprs = (sglQuotedString, dblQuotedString, quotedString, QuotedString('"', escQuote='""'), QuotedString("'", escQuote="''"), QuotedString('^'), QuotedString('<', endQuoteChar='>'))
        for expr in testExprs:
            strs = delimitedList(expr).searchString(src)
            print_(strs)
            self.assertTrue(bool(strs), "no matches found for test expression '%s'" % expr)
            for lst in strs:
                self.assertEqual(len(lst), 2, "invalid match found for test expression '%s'" % expr)
        from pyparsing import alphas, nums, Word
        src = "'ms1',1,0,'2009-12-22','2009-12-22 10:41:22') ON DUPLICATE KEY UPDATE sent_count = sent_count + 1, mtime = '2009-12-22 10:41:22';"
        tok_sql_quoted_value = QuotedString("'", '\\', "''", True, False) ^ QuotedString('"', '\\', '""', True, False)
        tok_sql_computed_value = Word(nums)
        tok_sql_identifier = Word(alphas)
        val = tok_sql_quoted_value | tok_sql_computed_value | tok_sql_identifier
        vals = delimitedList(val)
        print_(vals.parseString(src))
        self.assertEqual(len(vals.parseString(src)), 5, 'error in greedy quote escaping')