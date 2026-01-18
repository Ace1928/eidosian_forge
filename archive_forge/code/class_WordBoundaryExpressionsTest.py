from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WordBoundaryExpressionsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import WordEnd, WordStart, oneOf
        ws = WordStart()
        we = WordEnd()
        vowel = oneOf(list('AEIOUY'))
        consonant = oneOf(list('BCDFGHJKLMNPQRSTVWXZ'))
        leadingVowel = ws + vowel
        trailingVowel = vowel + we
        leadingConsonant = ws + consonant
        trailingConsonant = consonant + we
        internalVowel = ~ws + vowel + ~we
        bnf = leadingVowel | trailingVowel
        tests = '        ABC DEF GHI\n          JKL MNO PQR\n        STU VWX YZ  '.splitlines()
        tests.append('\n'.join(tests))
        expectedResult = [[['D', 'G'], ['A'], ['C', 'F'], ['I'], ['E'], ['A', 'I']], [['J', 'M', 'P'], [], ['L', 'R'], ['O'], [], ['O']], [['S', 'V'], ['Y'], ['X', 'Z'], ['U'], [], ['U', 'Y']], [['D', 'G', 'J', 'M', 'P', 'S', 'V'], ['A', 'Y'], ['C', 'F', 'L', 'R', 'X', 'Z'], ['I', 'O', 'U'], ['E'], ['A', 'I', 'O', 'U', 'Y']]]
        for t, expected in zip(tests, expectedResult):
            print_(t)
            results = [flatten(e.searchString(t).asList()) for e in [leadingConsonant, leadingVowel, trailingConsonant, trailingVowel, internalVowel, bnf]]
            print_(results)
            print_()
            self.assertEqual(results, expected, 'Failed WordBoundaryTest, expected %s, got %s' % (expected, results))