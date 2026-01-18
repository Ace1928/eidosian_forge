import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class testBoxUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the box uncertainty set (BoxSet).
    """

    def test_normal_construction_and_update(self):
        """
        Test BoxSet constructor and setter work normally
        when bounds are appropriate.
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)
        np.testing.assert_allclose(bounds, bset.bounds, err_msg='BoxSet bounds not as expected')
        new_bounds = [[3, 4], [5, 6]]
        bset.bounds = new_bounds
        np.testing.assert_allclose(new_bounds, bset.bounds, err_msg='BoxSet bounds not as expected')

    def test_error_on_box_set_dim_change(self):
        """
        BoxSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)
        exc_str = 'Attempting to set.*dimension 2 to a value of dimension 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = [[1, 2], [3, 4], [5, 6]]

    def test_error_on_lb_exceeds_ub(self):
        """
        Test exception raised when an LB exceeds a UB.
        """
        bad_bounds = [[1, 2], [4, 3]]
        exc_str = 'Lower bound 4 exceeds upper bound 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bad_bounds)
        bset = BoxSet([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = bad_bounds

    def test_error_on_ragged_bounds_array(self):
        """
        Test ValueError raised on attempting to set BoxSet bounds
        to a ragged array.

        This test also validates `uncertainty_sets.is_ragged` for all
        pre-defined array-like attributes of all set-types, as the
        `is_ragged` method is used throughout.
        """
        ragged_arrays = ([[1, 2], 3], [[1, 2], [3, [4, 5]]], [[1, 2], [3]])
        bset = BoxSet(bounds=[[1, 2], [3, 4]])
        exc_str = 'Argument `bounds` should not be a ragged array-like.*'
        for ragged_arr in ragged_arrays:
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bounds=ragged_arr)
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = ragged_arr

    def test_error_on_invalid_bounds_shape(self):
        """
        Test ValueError raised when attempting to set
        Box set bounds to array of incorrect shape
        (should be a 2-D array with 2 columns).
        """
        three_d_arr = [[[1, 2], [3, 4], [5, 6]]]
        exc_str = 'Argument `bounds` must be a 2-dimensional.*\\(detected 3 dimensions.*\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_d_arr)
        bset = BoxSet([[1, 2], [3, 4], [5, 6]])
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_d_arr

    def test_error_on_wrong_number_columns(self):
        """
        BoxSet bounds should be a 2D array-like with 2 columns.
        ValueError raised if number columns wrong
        """
        three_col_arr = [[1, 2, 3], [4, 5, 6]]
        exc_str = "Attribute 'bounds' should be of shape \\(\\.{3},2\\), but detected shape \\(\\.{3},3\\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_col_arr)
        bset = BoxSet([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_col_arr

    def test_error_on_empty_last_dimension(self):
        """
        Check ValueError raised when last dimension of BoxSet bounds is
        empty.
        """
        empty_2d_arr = [[], [], []]
        exc_str = 'Last dimension of argument `bounds` must be non-empty \\(detected shape \\(3, 0\\)\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bounds=empty_2d_arr)
        bset = BoxSet([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = empty_2d_arr

    def test_error_on_non_numeric_bounds(self):
        """
        Test that ValueError is raised if box set bounds
        are set to array-like with entries of a non-numeric
        type (such as int, float).
        """
        new_bounds = [[1, 'test'], [3, 2]]
        exc_str = "Entry 'test' of the argument `bounds` is not a valid numeric type \\(provided type 'str'\\)"
        with self.assertRaisesRegex(TypeError, exc_str):
            BoxSet(new_bounds)
        bset = BoxSet(bounds=[[1, 2], [3, 4]])
        with self.assertRaisesRegex(TypeError, exc_str):
            bset.bounds = new_bounds

    def test_error_on_bounds_with_nan_or_inf(self):
        """
        Box set bounds set to array-like with inf or nan.
        """
        bset = BoxSet(bounds=[[1, 2], [3, 4]])
        for val_str in ['inf', 'nan']:
            bad_bounds = [[1, float(val_str)], [2, 3]]
            exc_str = f"Entry '{val_str}' of the argument `bounds` is not a finite numeric value"
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bad_bounds)
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = bad_bounds

    def test_uncertainty_set_with_correct_params(self):
        """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list((id(u) for u in uncertain_params_in_expr)):
                        uncertain_params_in_expr.append(v)
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list((id(u) for u in vars_in_expr)):
                        vars_in_expr.append(v)
        self.assertEqual(len(vars_in_expr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        bounds = [(-1, 1), (-1, 1)]
        _set = BoxSet(bounds=bounds)
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the BoxSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)
        bounds = [(-1, 1), (-1, 1)]
        box_set = BoxSet(bounds=bounds)
        config = Block()
        config.uncertainty_set = box_set
        BoxSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertEqual(m.util.uncertain_param_vars[0].lb, -1, 'Bounds not added correctly for BoxSet')
        self.assertEqual(m.util.uncertain_param_vars[0].ub, 1, 'Bounds not added correctly for BoxSet')
        self.assertEqual(m.util.uncertain_param_vars[1].lb, -1, 'Bounds not added correctly for BoxSet')
        self.assertEqual(m.util.uncertain_param_vars[1].ub, 1, 'Bounds not added correctly for BoxSet')