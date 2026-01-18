from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class OptionalEachTest(ParseTestCase):

    def runTest1(self):
        from pyparsing import Optional, Keyword
        for the_input in ['Tal Weiss Major', 'Tal Major', 'Weiss Major', 'Major', 'Major Tal', 'Major Weiss', 'Major Tal Weiss']:
            print_(the_input)
            parser1 = Optional('Tal') + Optional('Weiss') & Keyword('Major')
            parser2 = Optional(Optional('Tal') + Optional('Weiss')) & Keyword('Major')
            p1res = parser1.parseString(the_input)
            p2res = parser2.parseString(the_input)
            self.assertEqual(p1res.asList(), p2res.asList(), 'Each failed to match with nested Optionals, ' + str(p1res.asList()) + ' should match ' + str(p2res.asList()))

    def runTest2(self):
        from pyparsing import Word, alphanums, OneOrMore, Group, Regex, Optional
        word = Word(alphanums + '_').setName('word')
        with_stmt = 'with' + OneOrMore(Group(word('key') + '=' + word('value')))('overrides')
        using_stmt = 'using' + Regex('id-[0-9a-f]{8}')('id')
        modifiers = Optional(with_stmt('with_stmt')) & Optional(using_stmt('using_stmt'))
        self.assertEqual(modifiers, 'with foo=bar bing=baz using id-deadbeef')
        self.assertNotEqual(modifiers, 'with foo=bar bing=baz using id-deadbeef using id-feedfeed')

    def runTest3(self):
        from pyparsing import Literal, Suppress, ZeroOrMore, OneOrMore
        foo = Literal('foo')
        bar = Literal('bar')
        openBrace = Suppress(Literal('{'))
        closeBrace = Suppress(Literal('}'))
        exp = openBrace + (OneOrMore(foo)('foo') & ZeroOrMore(bar)('bar')) + closeBrace
        tests = '            {foo}\n            {bar foo bar foo bar foo}\n            '.splitlines()
        for test in tests:
            test = test.strip()
            if not test:
                continue
            result = exp.parseString(test)
            print_(test, '->', result.asList())
            self.assertEqual(result.asList(), test.strip('{}').split(), 'failed to parse Each expression %r' % test)
            print_(result.dump())
        try:
            result = exp.parseString('{bar}')
            self.assertTrue(False, 'failed to raise exception when required element is missing')
        except ParseException as pe:
            pass

    def runTest4(self):
        from pyparsing import pyparsing_common, ZeroOrMore, Group
        expr = ~pyparsing_common.iso8601_date + pyparsing_common.integer('id') & ZeroOrMore(Group(pyparsing_common.iso8601_date)('date*'))
        expr.runTests('\n            1999-12-31 100 2001-01-01\n            42\n            ')

    def testParseExpressionsWithRegex(self):
        from itertools import product
        match_empty_regex = pp.Regex('[a-z]*')
        match_nonempty_regex = pp.Regex('[a-z]+')
        parser_classes = pp.ParseExpression.__subclasses__()
        test_string = 'abc def'
        expected = ['abc']
        for expr, cls in product((match_nonempty_regex, match_empty_regex), parser_classes):
            print_(expr, cls)
            parser = cls([expr])
            parsed_result = parser.parseString(test_string)
            print_(parsed_result.dump())
            self.assertParseResultsEquals(parsed_result, expected)
        for expr, cls in product((match_nonempty_regex, match_empty_regex), (pp.MatchFirst, pp.Or)):
            parser = cls([expr, expr])
            print_(parser)
            parsed_result = parser.parseString(test_string)
            print_(parsed_result.dump())
            self.assertParseResultsEquals(parsed_result, expected)

    def runTest(self):
        self.runTest1()
        self.runTest2()
        self.runTest3()
        self.runTest4()
        self.testParseExpressionsWithRegex()