from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ExprSplitterTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, quotedString, pythonStyleComment, Empty
        expr = Literal(';') + Empty()
        expr.ignore(quotedString)
        expr.ignore(pythonStyleComment)
        sample = '\n        def main():\n            this_semi_does_nothing();\n            neither_does_this_but_there_are_spaces_afterward();\n            a = "a;b"; return a # this is a comment; it has a semicolon!\n\n        def b():\n            if False:\n                z=1000;b("; in quotes");  c=200;return z\n            return \';\'\n\n        class Foo(object):\n            def bar(self):\n                \'\'\'a docstring; with a semicolon\'\'\'\n                a = 10; b = 11; c = 12\n\n                # this comment; has several; semicolons\n                if self.spam:\n                    x = 12; return x # so; does; this; one\n                    x = 15;;; y += x; return y\n\n            def baz(self):\n                return self.bar\n        '
        expected = [['            this_semi_does_nothing()', ''], ['            neither_does_this_but_there_are_spaces_afterward()', ''], ['            a = "a;b"', 'return a # this is a comment; it has a semicolon!'], ['                z=1000', 'b("; in quotes")', 'c=200', 'return z'], ["            return ';'"], ["                '''a docstring; with a semicolon'''"], ['                a = 10', 'b = 11', 'c = 12'], ['                # this comment; has several; semicolons'], ['                    x = 12', 'return x # so; does; this; one'], ['                    x = 15', '', '', 'y += x', 'return y']]
        exp_iter = iter(expected)
        for line in filter(lambda ll: ';' in ll, sample.splitlines()):
            print_(str(list(expr.split(line))) + ',')
            self.assertEqual(list(expr.split(line)), next(exp_iter), 'invalid split on expression')
        print_()
        expected = [['            this_semi_does_nothing()', ';', ''], ['            neither_does_this_but_there_are_spaces_afterward()', ';', ''], ['            a = "a;b"', ';', 'return a # this is a comment; it has a semicolon!'], ['                z=1000', ';', 'b("; in quotes")', ';', 'c=200', ';', 'return z'], ["            return ';'"], ["                '''a docstring; with a semicolon'''"], ['                a = 10', ';', 'b = 11', ';', 'c = 12'], ['                # this comment; has several; semicolons'], ['                    x = 12', ';', 'return x # so; does; this; one'], ['                    x = 15', ';', '', ';', '', ';', 'y += x', ';', 'return y']]
        exp_iter = iter(expected)
        for line in filter(lambda ll: ';' in ll, sample.splitlines()):
            print_(str(list(expr.split(line, includeSeparators=True))) + ',')
            self.assertEqual(list(expr.split(line, includeSeparators=True)), next(exp_iter), 'invalid split on expression')
        print_()
        expected = [['            this_semi_does_nothing()', ''], ['            neither_does_this_but_there_are_spaces_afterward()', ''], ['            a = "a;b"', 'return a # this is a comment; it has a semicolon!'], ['                z=1000', 'b("; in quotes");  c=200;return z'], ['                a = 10', 'b = 11; c = 12'], ['                    x = 12', 'return x # so; does; this; one'], ['                    x = 15', ';; y += x; return y']]
        exp_iter = iter(expected)
        for line in sample.splitlines():
            pieces = list(expr.split(line, maxsplit=1))
            print_(str(pieces) + ',')
            if len(pieces) == 2:
                exp = next(exp_iter)
                self.assertEqual(pieces, exp, 'invalid split on expression with maxSplits=1')
            elif len(pieces) == 1:
                self.assertEqual(len(expr.searchString(line)), 0, 'invalid split with maxSplits=1 when expr not present')
            else:
                print_('\n>>> ' + line)
                self.assertTrue(False, 'invalid split on expression with maxSplits=1, corner case')