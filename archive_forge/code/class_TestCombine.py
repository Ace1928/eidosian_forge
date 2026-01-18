from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestCombine(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Parsing real numbers - fail, parsed numbers are in pieces', expr=(pp.Word(pp.nums) + '.' + pp.Word(pp.nums))[...], text='1.2 2.3 3.1416 98.6', expected_list=['1', '.', '2', '2', '.', '3', '3', '.', '1416', '98', '.', '6']), PpTestSpec(desc='Parsing real numbers - better, use Combine to combine multiple tokens into one', expr=pp.Combine(pp.Word(pp.nums) + '.' + pp.Word(pp.nums))[...], text='1.2 2.3 3.1416 98.6', expected_list=['1.2', '2.3', '3.1416', '98.6'])]