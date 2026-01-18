from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class NestedExpressionsTest(ParseTestCase):

    def runTest(self):
        """
        This unit test checks nestedExpr in these ways:
        - use of default arguments
        - use of non-default arguments (such as a pyparsing-defined comment
          expression in place of quotedString)
        - use of a custom content expression
        - use of a pyparsing expression for opener and closer is *OPTIONAL*
        - use of input data containing nesting delimiters
        - correct grouping of parsed tokens according to nesting of opening
          and closing delimiters in the input string

        (Unit test written by christoph... as part of the Google Highly Open Participation Contest)
        """
        from pyparsing import nestedExpr, Literal, Regex, restOfLine, quotedString
        print_('Test defaults:')
        teststring = '(( ax + by)*C) (Z | (E^F) & D)'
        expr = nestedExpr()
        expected = [[['ax', '+', 'by'], '*C']]
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, "Defaults didn't work. That's a bad sign. Expected: %s, got: %s" % (expected, result))
        print_('\nNon-default opener')
        opener = '['
        teststring = test_string = '[[ ax + by)*C)'
        expected = [[['ax', '+', 'by'], '*C']]
        expr = nestedExpr('[')
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, "Non-default opener didn't work. Expected: %s, got: %s" % (expected, result))
        print_('\nNon-default closer')
        teststring = test_string = '(( ax + by]*C]'
        expected = [[['ax', '+', 'by'], '*C']]
        expr = nestedExpr(closer=']')
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, "Non-default closer didn't work. Expected: %s, got: %s" % (expected, result))
        print_('\nLiteral expressions for opener and closer')
        opener, closer = list(map(Literal, 'bar baz'.split()))
        expr = nestedExpr(opener, closer, content=Regex('([^b ]|b(?!a)|ba(?![rz]))+'))
        teststring = 'barbar ax + bybaz*Cbaz'
        expected = [[['ax', '+', 'by'], '*C']]
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, "Multicharacter opener and closer didn't work. Expected: %s, got: %s" % (expected, result))
        print_('\nUse ignore expression (1)')
        comment = Regex(';;.*')
        teststring = '\n        (let ((greeting "Hello, world!")) ;;(foo bar\n           (display greeting))\n        '
        expected = [['let', [['greeting', '"Hello,', 'world!"']], ';;(foo bar', ['display', 'greeting']]]
        expr = nestedExpr(ignoreExpr=comment)
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, 'Lisp-ish comments (";; <...> $") didn\'t work. Expected: %s, got: %s' % (expected, result))
        print_('\nUse ignore expression (2)')
        comment = ';;' + restOfLine
        teststring = '\n        (let ((greeting "Hello, )world!")) ;;(foo bar\n           (display greeting))\n        '
        expected = [['let', [['greeting', '"Hello, )world!"']], ';;', '(foo bar', ['display', 'greeting']]]
        expr = nestedExpr(ignoreExpr=comment ^ quotedString)
        result = expr.parseString(teststring)
        print_(result.dump())
        self.assertEqual(result.asList(), expected, 'Lisp-ish comments (";; <...> $") and quoted strings didn\'t work. Expected: %s, got: %s' % (expected, result))