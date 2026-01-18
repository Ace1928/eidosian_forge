from pyomo.common import unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.solver.sol_reader import parse_sol_file, SolFileData
class TestSolFileData(unittest.TestCase):

    def test_default_instantiation(self):
        instance = SolFileData()
        self.assertIsInstance(instance.primals, list)
        self.assertIsInstance(instance.duals, list)
        self.assertIsInstance(instance.var_suffixes, dict)
        self.assertIsInstance(instance.con_suffixes, dict)
        self.assertIsInstance(instance.obj_suffixes, dict)
        self.assertIsInstance(instance.problem_suffixes, dict)
        self.assertIsInstance(instance.other, list)