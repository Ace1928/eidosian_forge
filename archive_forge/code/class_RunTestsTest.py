from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RunTestsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, nums, delimitedList
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        intrange = integer('start') + '-' + integer('end')
        intrange.addCondition(lambda t: t.end > t.start, message='invalid range, start must be <= end', fatal=True)
        intrange.addParseAction(lambda t: list(range(t.start, t.end + 1)))
        indices = delimitedList(intrange | integer)
        indices.addParseAction(lambda t: sorted(set(t)))
        tests = '            # normal data\n            1-3,2-4,6,8-10,16\n\n            # lone integer\n            11'
        results = indices.runTests(tests, printResults=False)[1]
        expectedResults = [[1, 2, 3, 4, 6, 8, 9, 10, 16], [11]]
        for res, expected in zip(results, expectedResults):
            print_(res[1].asList())
            print_(expected)
            self.assertEqual(res[1].asList(), expected, 'failed test: ' + str(expected))
        tests = '            # invalid range\n            1-2, 3-1, 4-6, 7, 12\n            '
        success = indices.runTests(tests, printResults=False, failureTests=True)[0]
        self.assertTrue(success, 'failed to raise exception on improper range test')