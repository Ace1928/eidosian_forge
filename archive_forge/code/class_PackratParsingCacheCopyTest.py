from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PackratParsingCacheCopyTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, nums, delimitedList, Literal, Optional, alphas, alphanums, ZeroOrMore, empty
        integer = Word(nums).setName('integer')
        id = Word(alphas + '_', alphanums + '_')
        simpleType = Literal('int')
        arrayType = simpleType + ZeroOrMore('[' + delimitedList(integer) + ']')
        varType = arrayType | simpleType
        varDec = varType + delimitedList(id + Optional('=' + integer)) + ';'
        codeBlock = Literal('{}')
        funcDef = Optional(varType | 'void') + id + '(' + (delimitedList(varType + id) | 'void' | empty) + ')' + codeBlock
        program = varDec | funcDef
        input = 'int f(){}'
        results = program.parseString(input)
        print_("Parsed '%s' as %s" % (input, results.asList()))
        self.assertEqual(results.asList(), ['int', 'f', '(', ')', '{}'], 'Error in packrat parsing')