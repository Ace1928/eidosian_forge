import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
class TestSOS_noindex_024(SOSProblem_nonindexed, unittest.TestCase):

    def test(self):
        self.do_it(24)