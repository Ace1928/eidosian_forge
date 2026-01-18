from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WordExcludeTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, printables
        allButPunc = Word(printables, excludeChars='.,:;-_!?')
        test = "Hello, Mr. Ed, it's Wilbur!"
        result = allButPunc.searchString(test).asList()
        print_(result)
        self.assertEqual(result, [['Hello'], ['Mr'], ['Ed'], ["it's"], ['Wilbur']], 'failed WordExcludeTest')