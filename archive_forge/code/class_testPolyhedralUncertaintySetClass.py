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
class testPolyhedralUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the Polyhedral set.
    """

    def test_normal_construction_and_update(self):
        """
        Test PolyhedralSet constructor and attribute setters work
        appropriately.
        """
        lhs_coefficients_mat = [[1, 2, 3], [4, 5, 6]]
        rhs_vec = [1, 3]
        pset = PolyhedralSet(lhs_coefficients_mat, rhs_vec)
        np.testing.assert_allclose(lhs_coefficients_mat, pset.coefficients_mat)
        np.testing.assert_allclose(rhs_vec, pset.rhs_vec)
        pset.coefficients_mat = [[1, 0, 1], [1, 1, 1.5]]
        pset.rhs_vec = [3, 4]
        np.testing.assert_allclose([[1, 0, 1], [1, 1, 1.5]], pset.coefficients_mat)
        np.testing.assert_allclose([3, 4], pset.rhs_vec)

    def test_error_on_polyhedral_set_dim_change(self):
        """
        PolyhedralSet dimension (number columns of 'coefficients_mat')
        is considered immutable.
        Test ValueError raised if attempt made to change dimension.
        """
        pset = PolyhedralSet([[1, 2, 3], [4, 5, 6]], [1, 3])
        exc_str = '.*must have 3 columns to match set dimension \\(provided.*2 columns\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            pset.coefficients_mat = [[1, 2], [3, 4]]

    def test_error_on_inconsistent_rows(self):
        """
        Number of rows of budget membership mat is immutable.
        Similarly, size of rhs_vec is immutable.
        Check ValueError raised in event of attempted change.
        """
        coeffs_mat_exc_str = ".*must have 2 rows to match shape of attribute 'rhs_vec' \\(provided.*3 rows\\)"
        rhs_vec_exc_str = ".*must have 2 entries to match shape of attribute 'coefficients_mat' \\(provided.*3 entries\\)"
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3, 3])
        pset = PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3])
        with self.assertRaisesRegex(ValueError, coeffs_mat_exc_str):
            pset.coefficients_mat = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            pset.rhs_vec = [1, 3, 2]

    def test_error_on_empty_set(self):
        """
        Check ValueError raised if nonemptiness check performed
        at construction returns a negative result.
        """
        exc_str = 'PolyhedralSet.*is empty.*'
        with self.assertRaisesRegex(ValueError, exc_str):
            PolyhedralSet([[1], [-1]], rhs_vec=[1, -3])

    def test_error_on_polyhedral_mat_all_zero_columns(self):
        """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        rhs_vec = [1, 1, 2]
        exc_str = '.*all entries zero in columns at indexes: 0, 1.*'
        with self.assertRaisesRegex(ValueError, exc_str):
            PolyhedralSet(invalid_col_mat, rhs_vec)
        pset = PolyhedralSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], rhs_vec)
        with self.assertRaisesRegex(ValueError, exc_str):
            pset.coefficients_mat = invalid_col_mat

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
        A = [[0, 1], [1, 0]]
        b = [0, 0]
        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = ComponentSet()
        for con in m.uncertainty_set_contr.values():
            con_vars = ComponentSet(identify_variables(expr=con.expr))
            for v in m.uncertain_param_vars.values():
                if v in con_vars:
                    uncertain_params_in_expr.add(v)
        self.assertEqual(uncertain_params_in_expr, ComponentSet(m.uncertain_param_vars.values()), msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the PolyHedral is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        A = [[0, 1], [1, 0]]
        b = [0, 0]
        _set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend((v for v in m.uncertain_param_vars if v in ComponentSet(identify_variables(expr=con.expr))))
        self.assertEqual(len(vars_in_expr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_polyhedral_set_as_constraint(self):
        """
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        """
        A = [[1, 0], [0, 1]]
        b = [0, 0]
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        m.uncertainty_set_constr = polyhedral_set.set_as_constraint(uncertain_params=[m.p1, m.p2])
        self.assertEqual(len(A), len(m.uncertainty_set_constr.index_set()), msg='Polyhedral uncertainty set constraints must be as many as thenumber of rows in the matrix A.')

    def test_point_in_set(self):
        A = [[1, 0], [0, 1]]
        b = [0, 0]
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        self.assertTrue(polyhedral_set.point_in_set([0, 0]), msg='Point is not in the PolyhedralSet.')

    @unittest.skipUnless(baron_available, 'Global NLP solver is not available.')
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        A = [[1, 0], [0, 1]]
        b = [0, 0]
        polyhedral_set = PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)
        config = Block()
        config.uncertainty_set = polyhedral_set
        config.global_solver = SolverFactory('baron')
        PolyhedralSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for PolyhedralSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for PolyhedralSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for PolyhedralSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for PolyhedralSet')