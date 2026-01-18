import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
class TestSolverBase(unittest.TestCase):

    def test_abstract_member_list(self):
        expected_list = ['solve', 'available', 'version']
        member_list = list(base.SolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    def test_class_method_list(self):
        expected_list = ['Availability', 'CONFIG', 'available', 'is_persistent', 'solve', 'version']
        method_list = [method for method in dir(base.SolverBase) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_init(self):
        self.instance = base.SolverBase()
        self.assertFalse(self.instance.is_persistent())
        self.assertEqual(self.instance.version(), None)
        self.assertEqual(self.instance.name, 'solverbase')
        self.assertEqual(self.instance.CONFIG, self.instance.config)
        self.assertEqual(self.instance.solve(None), None)
        self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_context_manager(self):
        with base.SolverBase() as self.instance:
            self.assertFalse(self.instance.is_persistent())
            self.assertEqual(self.instance.version(), None)
            self.assertEqual(self.instance.name, 'solverbase')
            self.assertEqual(self.instance.CONFIG, self.instance.config)
            self.assertEqual(self.instance.solve(None), None)
            self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_config_kwds(self):
        self.instance = base.SolverBase(tee=True)
        self.assertTrue(self.instance.config.tee)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_availability(self):
        self.instance = base.SolverBase()
        self.instance.Availability._value_ = 1
        self.assertTrue(self.instance.Availability.__bool__(self.instance.Availability))
        self.instance.Availability._value_ = -1
        self.assertFalse(self.instance.Availability.__bool__(self.instance.Availability))

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_custom_solver_name(self):
        self.instance = base.SolverBase(name='my_unique_name')
        self.assertEqual(self.instance.name, 'my_unique_name')