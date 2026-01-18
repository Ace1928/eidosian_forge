from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class NestedAsDictTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, Forward, alphanums, Group, delimitedList, Dict, Word, Optional
        equals = Literal('=').suppress()
        lbracket = Literal('[').suppress()
        rbracket = Literal(']').suppress()
        lbrace = Literal('{').suppress()
        rbrace = Literal('}').suppress()
        value_dict = Forward()
        value_list = Forward()
        value_string = Word(alphanums + '@. ')
        value = value_list ^ value_dict ^ value_string
        values = Group(delimitedList(value, ','))
        value_list << lbracket + values + rbracket
        identifier = Word(alphanums + '_.')
        assignment = Group(identifier + equals + Optional(value))
        assignments = Dict(delimitedList(assignment, ';'))
        value_dict << lbrace + assignments + rbrace
        response = assignments
        rsp = 'username=goat; errors={username=[already taken, too short]}; empty_field='
        result_dict = response.parseString(rsp).asDict()
        print_(result_dict)
        self.assertEqual(result_dict['username'], 'goat', 'failed to process string in ParseResults correctly')
        self.assertEqual(result_dict['errors']['username'], ['already taken', 'too short'], 'failed to process nested ParseResults correctly')