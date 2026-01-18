from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseUsingRegex(ParseTestCase):

    def runTest(self):
        self.expect_warning = True
        import re
        signedInt = pp.Regex('[-+][0-9]+')
        unsignedInt = pp.Regex('[0-9]+')
        simpleString = pp.Regex('("[^\\"]*")|(\\\'[^\\\']*\\\')')
        namedGrouping = pp.Regex('("(?P<content>[^\\"]*)")')
        compiledRE = pp.Regex(re.compile('[A-Z]+'))

        def testMatch(expression, instring, shouldPass, expectedString=None):
            if shouldPass:
                try:
                    result = expression.parseString(instring)
                    print_('%s correctly matched %s' % (repr(expression), repr(instring)))
                    if expectedString != result[0]:
                        print_('\tbut failed to match the pattern as expected:')
                        print_('\tproduced %s instead of %s' % (repr(result[0]), repr(expectedString)))
                    return True
                except pp.ParseException:
                    print_('%s incorrectly failed to match %s' % (repr(expression), repr(instring)))
            else:
                try:
                    result = expression.parseString(instring)
                    print_('%s incorrectly matched %s' % (repr(expression), repr(instring)))
                    print_('\tproduced %s as a result' % repr(result[0]))
                except pp.ParseException:
                    print_('%s correctly failed to match %s' % (repr(expression), repr(instring)))
                    return True
            return False
        self.assertTrue(testMatch(signedInt, '1234 foo', False), 'Re: (1) passed, expected fail')
        self.assertTrue(testMatch(signedInt, '    +foo', False), 'Re: (2) passed, expected fail')
        self.assertTrue(testMatch(unsignedInt, 'abc', False), 'Re: (3) passed, expected fail')
        self.assertTrue(testMatch(unsignedInt, '+123 foo', False), 'Re: (4) passed, expected fail')
        self.assertTrue(testMatch(simpleString, 'foo', False), 'Re: (5) passed, expected fail')
        self.assertTrue(testMatch(simpleString, '"foo bar\'', False), 'Re: (6) passed, expected fail')
        self.assertTrue(testMatch(simpleString, '\'foo bar"', False), 'Re: (7) passed, expected fail')
        self.assertTrue(testMatch(signedInt, '   +123', True, '+123'), 'Re: (8) failed, expected pass')
        self.assertTrue(testMatch(signedInt, '+123', True, '+123'), 'Re: (9) failed, expected pass')
        self.assertTrue(testMatch(signedInt, '+123 foo', True, '+123'), 'Re: (10) failed, expected pass')
        self.assertTrue(testMatch(signedInt, '-0 foo', True, '-0'), 'Re: (11) failed, expected pass')
        self.assertTrue(testMatch(unsignedInt, '123 foo', True, '123'), 'Re: (12) failed, expected pass')
        self.assertTrue(testMatch(unsignedInt, '0 foo', True, '0'), 'Re: (13) failed, expected pass')
        self.assertTrue(testMatch(simpleString, '"foo"', True, '"foo"'), 'Re: (14) failed, expected pass')
        self.assertTrue(testMatch(simpleString, "'foo bar' baz", True, "'foo bar'"), 'Re: (15) failed, expected pass')
        self.assertTrue(testMatch(compiledRE, 'blah', False), 'Re: (16) passed, expected fail')
        self.assertTrue(testMatch(compiledRE, 'BLAH', True, 'BLAH'), 'Re: (17) failed, expected pass')
        self.assertTrue(testMatch(namedGrouping, '"foo bar" baz', True, '"foo bar"'), 'Re: (16) failed, expected pass')
        ret = namedGrouping.parseString('"zork" blah')
        print_(ret.asList())
        print_(list(ret.items()))
        print_(ret.content)
        self.assertEqual(ret.content, 'zork', 'named group lookup failed')
        self.assertEqual(ret[0], simpleString.parseString('"zork" blah')[0], 'Regex not properly returning ParseResults for named vs. unnamed groups')
        try:
            invRe = pp.Regex('("[^"]*")|(\'[^\']*\'')
        except Exception as e:
            print_('successfully rejected an invalid RE:', end=' ')
            print_(e)
        else:
            self.assertTrue(False, 'failed to reject invalid RE')
        invRe = pp.Regex('')