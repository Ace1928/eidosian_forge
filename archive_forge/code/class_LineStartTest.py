from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class LineStartTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        pass_tests = ['            AAA\n            BBB\n            ', '            AAA...\n            BBB\n            ']
        fail_tests = ['            AAA...\n            ...BBB\n            ', '            AAA  BBB\n            ']
        pass_tests = ['\n'.join((s.lstrip() for s in t.splitlines())).replace('.', ' ') for t in pass_tests]
        fail_tests = ['\n'.join((s.lstrip() for s in t.splitlines())).replace('.', ' ') for t in fail_tests]
        test_patt = pp.Word('A') - pp.LineStart() + pp.Word('B')
        print_(test_patt.streamline())
        success = test_patt.runTests(pass_tests)[0]
        self.assertTrue(success, 'failed LineStart passing tests (1)')
        success = test_patt.runTests(fail_tests, failureTests=True)[0]
        self.assertTrue(success, 'failed LineStart failure mode tests (1)')
        with AutoReset(pp.ParserElement, 'DEFAULT_WHITE_CHARS'):
            print_('no \\n in default whitespace chars')
            pp.ParserElement.setDefaultWhitespaceChars(' ')
            test_patt = pp.Word('A') - pp.LineStart() + pp.Word('B')
            print_(test_patt.streamline())
            success = test_patt.runTests(pass_tests, failureTests=True)[0]
            self.assertTrue(success, 'failed LineStart passing tests (2)')
            success = test_patt.runTests(fail_tests, failureTests=True)[0]
            self.assertTrue(success, 'failed LineStart failure mode tests (2)')
            test_patt = pp.Word('A') - pp.LineEnd().suppress() + pp.LineStart() + pp.Word('B') + pp.LineEnd().suppress()
            print_(test_patt.streamline())
            success = test_patt.runTests(pass_tests)[0]
            self.assertTrue(success, 'failed LineStart passing tests (3)')
            success = test_patt.runTests(fail_tests, failureTests=True)[0]
            self.assertTrue(success, 'failed LineStart failure mode tests (3)')
        test = '        AAA 1\n        AAA 2\n\n          AAA\n\n        B AAA\n\n        '
        from textwrap import dedent
        test = dedent(test)
        print_(test)
        for t, s, e in (pp.LineStart() + 'AAA').scanString(test):
            print_(s, e, pp.lineno(s, test), pp.line(s, test), ord(test[s]))
            print_()
            self.assertEqual(test[s], 'A', 'failed LineStart with insignificant newlines')
        with AutoReset(pp.ParserElement, 'DEFAULT_WHITE_CHARS'):
            pp.ParserElement.setDefaultWhitespaceChars(' ')
            for t, s, e in (pp.LineStart() + 'AAA').scanString(test):
                print_(s, e, pp.lineno(s, test), pp.line(s, test), ord(test[s]))
                print_()
                self.assertEqual(test[s], 'A', 'failed LineStart with insignificant newlines')