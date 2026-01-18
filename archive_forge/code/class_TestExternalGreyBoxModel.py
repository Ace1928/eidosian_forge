import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from ..pyomo_nlp import PyomoGreyBoxNLP
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
class TestExternalGreyBoxModel(unittest.TestCase):

    def test_pressure_drop_single_output(self):
        egbm = ex_models.PressureDropSingleOutput()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -36, -48], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_single_output_with_hessian(self):
        egbm = ex_models.PressureDropSingleOutputWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -36, -48], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()
        outputs_hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(outputs_hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(outputs_hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(outputs_hess.data, np.asarray([5 * (-8 * 3), 5 * (-8 * 2)], dtype=np.int64)))

    def test_pressure_drop_single_equality(self):
        egbm = ex_models.PressureDropSingleEquality()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])
        egbm.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([22], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 36, 48, 1], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            eq_hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_single_equality_with_hessian(self):
        egbm = ex_models.PressureDropSingleEqualityWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])
        egbm.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([5], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([22], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 36, 48, 1], dtype=np.float64)))
        eq_hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            outputs_hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(eq_hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(eq_hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(eq_hess.data, np.asarray([5 * (8 * 3), 5 * (8 * 2)], dtype=np.float64)))

    def test_pressure_drop_two_outputs(self):
        egbm = ex_models.PressureDropTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual([], eq_con_names)
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([3.0, 5.0], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([64, 28], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -18, -24, 1, -36, -48], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            hess_eq = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_two_outputs_with_hessian(self):
        egbm = ex_models.PressureDropTwoOutputsWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual([], eq_con_names)
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([], dtype=np.float64))
        with self.assertRaises(AssertionError):
            egbm.set_equality_constraint_multipliers(np.asarray([1], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([3.0, 5.0], dtype=np.float64))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([64, 28], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -18, -24, 1, -36, -48], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            hess_eq = egbm.evaluate_hessian_equality_constraints()
        hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.data, np.asarray([-156.0, -104.0], dtype=np.float64)))

    def test_pressure_drop_two_equalities(self):
        egbm = ex_models.PressureDropTwoEqualities()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
        output_names = egbm.output_names()
        self.assertEqual([], output_names)
        egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([3, 5], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([]))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()
        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_equality_constraints()

    def test_pressure_drop_two_equalities_with_hessian(self):
        egbm = ex_models.PressureDropTwoEqualitiesWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
        output_names = egbm.output_names()
        self.assertEqual([], output_names)
        egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([3, 5], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([]))
        with self.assertRaises(AssertionError):
            egbm.set_output_constraint_multipliers(np.asarray([1], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()
        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            hess_outputs = egbm.evaluate_hessian_outputs()
        hess = egbm.evaluate_hessian_equality_constraints()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.data, np.asarray([96.0, 64.0], dtype=np.float64)))

    def test_pressure_drop_two_equalities_two_outputs(self):
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)))
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)))
        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_equality_constraints()
        with self.assertRaises(AttributeError):
            hess = egbm.evaluate_hessian_outputs()

    def test_pressure_drop_two_equalities_two_outputs_with_hessian(self):
        egbm = ex_models.PressureDropTwoEqualitiesTwoOutputsWithHessian()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])
        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        egbm.set_equality_constraint_multipliers(np.asarray([2, 4], dtype=np.float64))
        egbm.set_output_constraint_multipliers(np.asarray([7, 9], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))
        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)))
        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)))
        hess = egbm.evaluate_hessian_equality_constraints()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.data, np.asarray([60.0, 40.0], dtype=np.float64)))
        hess = egbm.evaluate_hessian_outputs()
        self.assertTrue(np.array_equal(hess.row, np.asarray([2, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.col, np.asarray([1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(hess.data, np.asarray([-258, -172], dtype=np.float64)))