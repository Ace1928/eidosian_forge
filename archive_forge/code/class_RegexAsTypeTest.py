from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RegexAsTypeTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        test_str = 'sldkjfj 123 456 lsdfkj'
        print_('return as list of match groups')
        expr = pp.Regex('\\w+ (\\d+) (\\d+) (\\w+)', asGroupList=True)
        expected_group_list = [tuple(test_str.split()[1:])]
        result = expr.parseString(test_str)
        print_(result.dump())
        print_(expected_group_list)
        self.assertEqual(result.asList(), expected_group_list, 'incorrect group list returned by Regex)')
        print_('return as re.match instance')
        expr = pp.Regex('\\w+ (?P<num1>\\d+) (?P<num2>\\d+) (?P<last_word>\\w+)', asMatch=True)
        result = expr.parseString(test_str)
        print_(result.dump())
        print_(result[0].groups())
        print_(expected_group_list)
        self.assertEqual(result[0].groupdict(), {'num1': '123', 'num2': '456', 'last_word': 'lsdfkj'}, 'invalid group dict from Regex(asMatch=True)')
        self.assertEqual(result[0].groups(), expected_group_list[0], 'incorrect group list returned by Regex(asMatch)')