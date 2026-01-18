import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptSolverConfig(unittest.TestCase):

    def test_default_instantiation(self):
        config = ipopt.IpoptConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)
        self.assertIsInstance(config.executable, type(Executable('path')))
        self.assertIsInstance(config.writer_config, type(NLWriter.CONFIG()))

    def test_custom_instantiation(self):
        config = ipopt.IpoptConfig(description='A description')
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, 'A description')
        self.assertIsNone(config.time_limit)
        self.assertIsNotNone(str(config.executable))
        self.assertIn('ipopt', str(config.executable))
        config.executable = Executable('/bogus/path')
        self.assertIsNone(config.executable.executable)
        self.assertFalse(config.executable.available())