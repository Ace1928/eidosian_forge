from pyomo.common import unittest
from pyomo.contrib.solver.config import (
class TestBranchAndBoundConfig(unittest.TestCase):

    def test_interface_default_instantiation(self):
        config = BranchAndBoundConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.rel_gap)
        self.assertIsNone(config.abs_gap)

    def test_interface_custom_instantiation(self):
        config = BranchAndBoundConfig(description='A description')
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, 'A description')
        self.assertFalse(config.time_limit)
        config.time_limit = 1.0
        self.assertEqual(config.time_limit, 1.0)
        self.assertIsInstance(config.time_limit, float)
        config.rel_gap = 2.5
        self.assertEqual(config.rel_gap, 2.5)