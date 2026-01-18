from pyomo.common import unittest
from pyomo.contrib.solver.config import (
class TestAutoUpdateConfig(unittest.TestCase):

    def test_interface_default_instantiation(self):
        config = AutoUpdateConfig()
        self.assertTrue(config.check_for_new_or_removed_constraints)
        self.assertTrue(config.check_for_new_or_removed_vars)
        self.assertTrue(config.check_for_new_or_removed_params)
        self.assertTrue(config.check_for_new_objective)
        self.assertTrue(config.update_constraints)
        self.assertTrue(config.update_vars)
        self.assertTrue(config.update_named_expressions)
        self.assertTrue(config.update_objective)
        self.assertTrue(config.update_objective)
        self.assertTrue(config.treat_fixed_vars_as_params)

    def test_interface_custom_instantiation(self):
        config = AutoUpdateConfig(description='A description')
        config.check_for_new_objective = False
        self.assertEqual(config._description, 'A description')
        self.assertTrue(config.check_for_new_or_removed_constraints)
        self.assertFalse(config.check_for_new_objective)