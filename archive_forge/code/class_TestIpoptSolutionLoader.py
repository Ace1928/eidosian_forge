import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
class TestIpoptSolutionLoader(unittest.TestCase):

    def test_get_reduced_costs_error(self):
        loader = ipopt.IpoptSolutionLoader(None, None)
        with self.assertRaises(RuntimeError):
            loader.get_reduced_costs()

        class NLInfo:
            pass
        loader._nl_info = NLInfo()
        loader._nl_info.eliminated_vars = [1, 2, 3]
        with self.assertRaises(NotImplementedError):
            loader.get_reduced_costs()
        loader._nl_info.eliminated_vars = []
        with self.assertRaises(DeveloperError):
            loader.get_reduced_costs()