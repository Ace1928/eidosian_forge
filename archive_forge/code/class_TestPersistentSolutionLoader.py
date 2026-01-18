from pyomo.common import unittest
from pyomo.contrib.solver.solution import SolutionLoaderBase, PersistentSolutionLoader
class TestPersistentSolutionLoader(unittest.TestCase):

    def test_abstract_member_list(self):
        member_list = list(PersistentSolutionLoader('ipopt').__abstractmethods__)
        self.assertEqual(member_list, [])

    def test_member_list(self):
        expected_list = ['load_vars', 'get_primals', 'get_duals', 'get_reduced_costs', 'invalidate']
        method_list = [method for method in dir(PersistentSolutionLoader) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_initialization(self):
        self.instance = PersistentSolutionLoader('ipopt')
        self.assertTrue(self.instance._valid)
        self.assertEqual(self.instance._solver, 'ipopt')

    def test_invalid(self):
        self.instance = PersistentSolutionLoader('ipopt')
        self.instance.invalidate()
        with self.assertRaises(RuntimeError):
            self.instance.get_primals()