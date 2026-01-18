import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
class TestSuffixFinder(unittest.TestCase):

    def test_suffix_finder(self):
        m = ConcreteModel()
        m.v1 = Var()
        m.b1 = Block()
        m.b1.v2 = Var()
        m.b1.b2 = Block()
        m.b1.b2.v3 = Var([0])
        _suffix_finder = SuffixFinder('suffix')
        m.suffix = Suffix(direction=Suffix.EXPORT)
        m.b1.b2.suffix = Suffix(direction=Suffix.EXPORT)
        assert _suffix_finder.find(m.b1.b2.v3[0]) == None
        m.suffix[None] = 1
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 1
        m.b1.b2.suffix[None] = 2
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 2
        m.b1.b2.suffix[m.b1.b2.v3] = 3
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 3
        m.suffix[m.b1.b2.v3] = 4
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 4
        m.b1.b2.suffix[m.b1.b2.v3[0]] = 5
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 5
        m.suffix[m.b1.b2.v3[0]] = 6
        assert _suffix_finder.find(m.b1.b2.v3[0]) == 6
        assert _suffix_finder.find(m.b1.v2) == 1
        m.b1.b2.suffix[m.v1] = 5
        assert _suffix_finder.find(m.v1) == 1