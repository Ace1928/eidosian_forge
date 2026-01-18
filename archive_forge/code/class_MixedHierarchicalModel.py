from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
class MixedHierarchicalModel(object):

    def __init__(self):
        m = self.model = ConcreteModel()
        m.a = Block()
        m.a.c = DerivedBlock()
        m.b = DerivedBlock()
        m.b.d = DerivedBlock()
        m.b.e = Block()
        m.b.e.f = DerivedBlock()
        m.b.e.f.g = Block()
        self.PrefixDFS_block = ['unknown', 'a']
        self.PostfixDFS_block = ['a', 'unknown']
        self.BFS_block = ['unknown', 'a']
        self.PrefixDFS_both = ['unknown', 'a', 'a.c', 'b', 'b.d', 'b.e', 'b.e.f', 'b.e.f.g']
        self.PostfixDFS_both = ['a.c', 'a', 'b.d', 'b.e.f.g', 'b.e.f', 'b.e', 'b', 'unknown']
        self.BFS_both = ['unknown', 'a', 'b', 'a.c', 'b.d', 'b.e', 'b.e.f', 'b.e.f.g']
        self.PrefixDFS_block_subclass = ['a', 'b.e', 'b.e.f.g']
        self.PostfixDFS_block_subclass = ['b.e.f.g', 'b.e', 'a']
        self.BFS_block_subclass = ['a', 'b.e', 'b.e.f.g']