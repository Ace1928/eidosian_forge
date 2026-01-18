import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
class TestSOS_indexedmulti_000(SOSProblem_indexedmultivar, unittest.TestCase):

    def test(self):
        self.do_it(0)