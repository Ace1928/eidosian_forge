from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestResultsName(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Match with results name', expr=pp.Literal('xyz').setResultsName('value'), text='xyz', expected_dict={'value': 'xyz'}, expected_list=['xyz']), PpTestSpec(desc='Match with results name - using naming short-cut', expr=pp.Literal('xyz')('value'), text='xyz', expected_dict={'value': 'xyz'}, expected_list=['xyz']), PpTestSpec(desc='Define multiple results names', expr=pp.Word(pp.alphas, pp.alphanums)('key') + '=' + pp.pyparsing_common.integer('value'), text='range=5280', expected_dict={'key': 'range', 'value': 5280}, expected_list=['range', '=', 5280])]