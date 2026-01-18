import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
class CPLEXShellWritePrioritiesFile(unittest.TestCase):
    """Unit test on writing of priorities via `CPLEXSHELL._write_priorities_file()`"""
    suffix_cls = Suffix

    def setUp(self):
        TempfileManager.push()
        self.mock_model = self.get_mock_model()
        self.mock_cplex_shell = self.get_mock_cplex_shell(self.mock_model)
        self.mock_cplex_shell._priorities_file_name = TempfileManager.create_tempfile(suffix='.cplex.ord')

    def tearDown(self):
        TempfileManager.clear_tempfiles()

    def get_mock_model(self):
        model = ConcreteModel()
        model.x = Var(within=Binary)
        model.con = Constraint(expr=model.x >= 1)
        model.obj = Objective(expr=model.x)
        return model

    def get_mock_cplex_shell(self, mock_model):
        solver = MockCPLEX()
        solver._problem_files, solver._problem_format, solver._smap_id = convert_problem((mock_model,), ProblemFormat.cpxlp, [ProblemFormat.cpxlp], has_capability=lambda x: True, symbolic_solver_labels=True)
        return solver

    def get_priorities_file_as_string(self, mock_cplex_shell):
        with open(mock_cplex_shell._priorities_file_name, 'r') as ord_file:
            priorities_file = ord_file.read()
        return priorities_file

    @staticmethod
    def _set_suffix_value(suffix, variable, value):
        suffix.set_value(variable, value)

    def test_write_without_priority_suffix(self):
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_write_priority_to_priorities_file(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)
        self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x 10\nENDATA\n')

    def test_write_priority_and_direction_to_priorities_file(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)
        self.mock_model.direction = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        direction_val = BranchDirection.down
        self._set_suffix_value(self.mock_model.direction, self.mock_model.x, direction_val)
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)
        self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n DN x 10\nENDATA\n')

    def test_raise_due_to_invalid_priority(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, -1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, 1.1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_use_default_due_to_invalid_direction(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)
        self.mock_model.direction = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self._set_suffix_value(self.mock_model.direction, self.mock_model.x, 'invalid_branching_direction')
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)
        self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x 10\nENDATA\n')