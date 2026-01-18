from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestParseCondition(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Define a condition to only match numeric values that are multiples of 7', expr=pp.Word(pp.nums).addCondition(lambda t: int(t[0]) % 7 == 0)[...], text='14 35 77 12 28', expected_list=['14', '35', '77']), PpTestSpec(desc='Separate conversion to int and condition into separate parse action/conditions', expr=pp.Word(pp.nums).addParseAction(lambda t: int(t[0])).addCondition(lambda t: t[0] % 7 == 0)[...], text='14 35 77 12 28', expected_list=[14, 35, 77])]