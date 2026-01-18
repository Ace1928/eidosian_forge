import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
class TestAmplNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pm2 = create_pyomo_model2()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, 'Pyomo_TestAmplNLP')
        cls.pm2.write(cls.filename + '.nl', io_options={'symbolic_solver_labels': True})
        cls.nlp = AmplNLP(cls.filename + '.nl', row_filename=cls.filename + '.row', col_filename=cls.filename + '.col')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_names(self):
        expected_variable_names = ['x[1]', 'x[2]', 'x[3]']
        variable_names = self.nlp.variable_names()
        self.assertEqual(len(expected_variable_names), len(variable_names))
        for i in range(len(expected_variable_names)):
            self.assertTrue(expected_variable_names[i] in variable_names)
        expected_constraint_names = ['e1', 'e2', 'i1', 'i2', 'i3']
        constraint_names = self.nlp.constraint_names()
        self.assertEqual(len(expected_constraint_names), len(constraint_names))
        for i in range(len(expected_constraint_names)):
            self.assertTrue(expected_constraint_names[i] in constraint_names)
        expected_eq_constraint_names = ['e1', 'e2']
        eq_constraint_names = self.nlp.eq_constraint_names()
        self.assertEqual(len(expected_eq_constraint_names), len(eq_constraint_names))
        for i in range(len(expected_eq_constraint_names)):
            self.assertTrue(expected_eq_constraint_names[i] in eq_constraint_names)
        expected_ineq_constraint_names = ['i1', 'i2', 'i3']
        ineq_constraint_names = self.nlp.ineq_constraint_names()
        self.assertEqual(len(expected_ineq_constraint_names), len(ineq_constraint_names))
        for i in range(len(expected_ineq_constraint_names)):
            self.assertTrue(expected_ineq_constraint_names[i] in ineq_constraint_names)

    def test_idxs(self):
        variable_idxs = list()
        variable_idxs.append(self.nlp.variable_idx('x[1]'))
        variable_idxs.append(self.nlp.variable_idx('x[2]'))
        variable_idxs.append(self.nlp.variable_idx('x[3]'))
        self.assertEqual(sum(variable_idxs), 3)
        constraint_idxs = list()
        constraint_idxs.append(self.nlp.constraint_idx('e1'))
        constraint_idxs.append(self.nlp.constraint_idx('e2'))
        constraint_idxs.append(self.nlp.constraint_idx('i1'))
        constraint_idxs.append(self.nlp.constraint_idx('i2'))
        constraint_idxs.append(self.nlp.constraint_idx('i3'))
        self.assertEqual(sum(constraint_idxs), 10)
        eq_constraint_idxs = list()
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('e1'))
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('e2'))
        self.assertEqual(sum(eq_constraint_idxs), 1)
        ineq_constraint_idxs = list()
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i1'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i2'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i3'))
        self.assertEqual(sum(ineq_constraint_idxs), 3)