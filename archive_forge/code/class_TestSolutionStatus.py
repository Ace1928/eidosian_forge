from io import StringIO
from typing import Sequence, Dict, Optional, Mapping, MutableMapping
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver import results
from pyomo.contrib.solver import solution
import pyomo.environ as pyo
from pyomo.core.base.var import Var
class TestSolutionStatus(unittest.TestCase):

    def test_member_list(self):
        member_list = results.SolutionStatus._member_names_
        expected_list = ['noSolution', 'infeasible', 'feasible', 'optimal']
        self.assertEqual(member_list, expected_list)

    def test_codes(self):
        self.assertEqual(results.SolutionStatus.noSolution.value, 0)
        self.assertEqual(results.SolutionStatus.infeasible.value, 10)
        self.assertEqual(results.SolutionStatus.feasible.value, 20)
        self.assertEqual(results.SolutionStatus.optimal.value, 30)