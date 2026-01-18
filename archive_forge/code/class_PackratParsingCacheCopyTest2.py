from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PackratParsingCacheCopyTest2(ParseTestCase):

    def runTest(self):
        from pyparsing import Keyword, Word, Suppress, Forward, Optional, delimitedList, Group
        DO, AA = list(map(Keyword, 'DO AA'.split()))
        LPAR, RPAR = list(map(Suppress, '()'))
        identifier = ~AA + Word('Z')
        function_name = identifier.copy()
        expr = Forward().setName('expr')
        expr << (Group(function_name + LPAR + Optional(delimitedList(expr)) + RPAR).setName('functionCall') | identifier.setName('ident'))
        stmt = DO + Group(delimitedList(identifier + '.*' | expr))
        result = stmt.parseString('DO Z')
        print_(result.asList())
        self.assertEqual(len(result[1]), 1, 'packrat parsing is duplicating And term exprs')