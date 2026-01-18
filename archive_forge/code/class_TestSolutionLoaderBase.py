from pyomo.common import unittest
from pyomo.contrib.solver.solution import SolutionLoaderBase, PersistentSolutionLoader
class TestSolutionLoaderBase(unittest.TestCase):

    def test_abstract_member_list(self):
        expected_list = ['get_primals']
        member_list = list(SolutionLoaderBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    def test_member_list(self):
        expected_list = ['load_vars', 'get_primals', 'get_duals', 'get_reduced_costs']
        method_list = [method for method in dir(SolutionLoaderBase) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    @unittest.mock.patch.multiple(SolutionLoaderBase, __abstractmethods__=set())
    def test_solution_loader_base(self):
        self.instance = SolutionLoaderBase()
        self.assertEqual(self.instance.get_primals(), None)
        with self.assertRaises(NotImplementedError):
            self.instance.get_duals()
        with self.assertRaises(NotImplementedError):
            self.instance.get_reduced_costs()