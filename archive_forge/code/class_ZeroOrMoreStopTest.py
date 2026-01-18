from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ZeroOrMoreStopTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, ZeroOrMore, alphas, Keyword, CaselessKeyword
        test = 'BEGIN END'
        BEGIN, END = map(Keyword, 'BEGIN,END'.split(','))
        body_word = Word(alphas).setName('word')
        for ender in (END, 'END', CaselessKeyword('END')):
            expr = BEGIN + ZeroOrMore(body_word, stopOn=ender) + END
            self.assertEqual(test, expr, 'Did not successfully stop on ending expression %r' % ender)
            if PY_3:
                expr = eval('BEGIN + body_word[0, ...].stopOn(ender) + END')
                self.assertEqual(test, expr, 'Did not successfully stop on ending expression %r' % ender)