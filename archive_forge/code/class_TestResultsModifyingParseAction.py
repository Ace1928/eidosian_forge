from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestResultsModifyingParseAction(PyparsingExpressionTestCase):

    def compute_stats_parse_action(t):
        t['sum'] = sum(t)
        t['ave'] = sum(t) / len(t)
        t['min'] = min(t)
        t['max'] = max(t)
    tests = [PpTestSpec(desc='A parse action that adds new key-values', expr=pp.pyparsing_common.integer[...].addParseAction(compute_stats_parse_action), text='27 1 14 22 89', expected_list=[27, 1, 14, 22, 89], expected_dict={'ave': 30.6, 'max': 89, 'min': 1, 'sum': 153})]