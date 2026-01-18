from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class DefaultKeywordCharsTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        try:
            pp.Keyword('start').parseString('start1000')
        except pp.ParseException:
            pass
        else:
            self.assertTrue(False, 'failed to fail on default keyword chars')
        try:
            pp.Keyword('start', identChars=pp.alphas).parseString('start1000')
        except pp.ParseException:
            self.assertTrue(False, 'failed to match keyword using updated keyword chars')
        else:
            pass
        with AutoReset(pp.Keyword, 'DEFAULT_KEYWORD_CHARS'):
            pp.Keyword.setDefaultKeywordChars(pp.alphas)
            try:
                pp.Keyword('start').parseString('start1000')
            except pp.ParseException:
                self.assertTrue(False, 'failed to match keyword using updated keyword chars')
            else:
                pass
        try:
            pp.CaselessKeyword('START').parseString('start1000')
        except pp.ParseException:
            pass
        else:
            self.assertTrue(False, 'failed to fail on default keyword chars')
        try:
            pp.CaselessKeyword('START', identChars=pp.alphas).parseString('start1000')
        except pp.ParseException:
            self.assertTrue(False, 'failed to match keyword using updated keyword chars')
        else:
            pass
        with AutoReset(pp.Keyword, 'DEFAULT_KEYWORD_CHARS'):
            pp.Keyword.setDefaultKeywordChars(pp.alphas)
            try:
                pp.CaselessKeyword('START').parseString('start1000')
            except pp.ParseException:
                self.assertTrue(False, 'failed to match keyword using updated keyword chars')
            else:
                pass