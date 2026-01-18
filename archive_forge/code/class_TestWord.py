from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestWord(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Simple Word match', expr=pp.Word('xy'), text='xxyxxyy', expected_list=['xxyxxyy']), PpTestSpec(desc='Simple Word match of two separate Words', expr=pp.Word('x') + pp.Word('y'), text='xxxxxyy', expected_list=['xxxxx', 'yy']), PpTestSpec(desc='Simple Word match of two separate Words - implicitly skips whitespace', expr=pp.Word('x') + pp.Word('y'), text='xxxxx yy', expected_list=['xxxxx', 'yy'])]