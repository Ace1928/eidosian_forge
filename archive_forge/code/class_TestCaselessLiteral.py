from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestCaselessLiteral(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Match colors, converting to consistent case', expr=(pp.CaselessLiteral('RED') | pp.CaselessLiteral('GREEN') | pp.CaselessLiteral('BLUE'))[...], text='red Green BluE blue GREEN green rEd', expected_list=['RED', 'GREEN', 'BLUE', 'BLUE', 'GREEN', 'GREEN', 'RED'])]