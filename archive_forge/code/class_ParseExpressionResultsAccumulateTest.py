from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseExpressionResultsAccumulateTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, delimitedList, Combine, alphas, nums
        num = Word(nums).setName('num')('base10*')
        hexnum = Combine('0x' + Word(nums)).setName('hexnum')('hex*')
        name = Word(alphas).setName('word')('word*')
        list_of_num = delimitedList(hexnum | num | name, ',')
        tokens = list_of_num.parseString('1, 0x2, 3, 0x4, aaa')
        for k, llen, lst in (('base10', 2, ['1', '3']), ('hex', 2, ['0x2', '0x4']), ('word', 1, ['aaa'])):
            print_(k, tokens[k])
            self.assertEqual(len(tokens[k]), llen, 'Wrong length for key %s, %s' % (k, str(tokens[k].asList())))
            self.assertEqual(lst, tokens[k].asList(), 'Incorrect list returned for key %s, %s' % (k, str(tokens[k].asList())))
        self.assertEqual(tokens.base10.asList(), ['1', '3'], 'Incorrect list for attribute base10, %s' % str(tokens.base10.asList()))
        self.assertEqual(tokens.hex.asList(), ['0x2', '0x4'], 'Incorrect list for attribute hex, %s' % str(tokens.hex.asList()))
        self.assertEqual(tokens.word.asList(), ['aaa'], 'Incorrect list for attribute word, %s' % str(tokens.word.asList()))
        from pyparsing import Literal, Word, nums, Group, Dict, alphas, quotedString, oneOf, delimitedList, removeQuotes, alphanums
        lbrack = Literal('(').suppress()
        rbrack = Literal(')').suppress()
        integer = Word(nums).setName('int')
        variable = Word(alphas, max=1).setName('variable')
        relation_body_item = variable | integer | quotedString.copy().setParseAction(removeQuotes)
        relation_name = Word(alphas + '_', alphanums + '_')
        relation_body = lbrack + Group(delimitedList(relation_body_item)) + rbrack
        Goal = Dict(Group(relation_name + relation_body))
        Comparison_Predicate = Group(variable + oneOf('< >') + integer)('pred*')
        Query = Goal('head') + ':-' + delimitedList(Goal | Comparison_Predicate)
        test = 'Q(x,y,z):-Bloo(x,"Mitsis",y),Foo(y,z,1243),y>28,x<12,x>3'
        queryRes = Query.parseString(test)
        print_('pred', queryRes.pred)
        self.assertEqual(queryRes.pred.asList(), [['y', '>', '28'], ['x', '<', '12'], ['x', '>', '3']], 'Incorrect list for attribute pred, %s' % str(queryRes.pred.asList()))
        print_(queryRes.dump())