from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RegexSubTest(ParseTestCase):

    def runTest(self):
        self.expect_warning = True
        import pyparsing as pp
        print_('test sub with string')
        expr = pp.Regex('<title>').sub("'Richard III'")
        result = expr.transformString('This is the title: <title>')
        print_(result)
        self.assertEqual(result, "This is the title: 'Richard III'", 'incorrect Regex.sub result with simple string')
        print_('test sub with re string')
        expr = pp.Regex('([Hh]\\d):\\s*(.*)').sub('<\\1>\\2</\\1>')
        result = expr.transformString('h1: This is the main heading\nh2: This is the sub-heading')
        print_(result)
        self.assertEqual(result, '<h1>This is the main heading</h1>\n<h2>This is the sub-heading</h2>', 'incorrect Regex.sub result with re string')
        print_('test sub with re string (Regex returns re.match)')
        expr = pp.Regex('([Hh]\\d):\\s*(.*)', asMatch=True).sub('<\\1>\\2</\\1>')
        result = expr.transformString('h1: This is the main heading\nh2: This is the sub-heading')
        print_(result)
        self.assertEqual(result, '<h1>This is the main heading</h1>\n<h2>This is the sub-heading</h2>', 'incorrect Regex.sub result with re string')
        print_('test sub with callable that return str')
        expr = pp.Regex('<(.*?)>').sub(lambda m: m.group(1).upper())
        result = expr.transformString('I want this in upcase: <what? what?>')
        print_(result)
        self.assertEqual(result, 'I want this in upcase: WHAT? WHAT?', 'incorrect Regex.sub result with callable')
        try:
            expr = pp.Regex('<(.*?)>', asMatch=True).sub(lambda m: m.group(1).upper())
        except SyntaxError:
            pass
        else:
            self.assertTrue(False, 'failed to warn using a Regex.sub(callable) with asMatch=True')
        try:
            expr = pp.Regex('<(.*?)>', asGroupList=True).sub(lambda m: m.group(1).upper())
        except SyntaxError:
            pass
        else:
            self.assertTrue(False, 'failed to warn using a Regex.sub() with asGroupList=True')
        try:
            expr = pp.Regex('<(.*?)>', asGroupList=True).sub('')
        except SyntaxError:
            pass
        else:
            self.assertTrue(False, 'failed to warn using a Regex.sub() with asGroupList=True')