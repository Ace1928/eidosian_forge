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
class testAxisAlignedEllipsoidalUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the AxisAlignedEllipsoidalSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test AxisAlignedEllipsoidalSet constructor and setter
        work normally when bounds are appropriate.
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)
        np.testing.assert_allclose(center, aset.center, err_msg='AxisAlignedEllipsoidalSet center not as expected')
        np.testing.assert_allclose(half_lengths, aset.half_lengths, err_msg='AxisAlignedEllipsoidalSet half-lengths not as expected')
        new_center = [-1, -3]
        new_half_lengths = [0, 1]
        aset.center = new_center
        aset.half_lengths = new_half_lengths
        np.testing.assert_allclose(new_center, aset.center, err_msg='AxisAlignedEllipsoidalSet center update not as expected')
        np.testing.assert_allclose(new_half_lengths, aset.half_lengths, err_msg='AxisAlignedEllipsoidalSet half lengths update not as expected')

    def test_error_on_axis_aligned_dim_change(self):
        """
        AxisAlignedEllipsoidalSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)
        exc_str = 'Attempting to set.*dimension 2 to value of dimension 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.center = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.half_lengths = [0, 0, 1]

    def test_error_on_negative_axis_aligned_half_lengths(self):
        """
        Test ValueError if half lengths for AxisAlignedEllipsoidalSet
        contains a negative value.
        """
        center = [1, 1]
        invalid_half_lengths = [1, -1]
        exc_str = "Entry -1 of.*'half_lengths' is negative.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            AxisAlignedEllipsoidalSet(center, invalid_half_lengths)
        aset = AxisAlignedEllipsoidalSet(center, [1, 0])
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.half_lengths = invalid_half_lengths

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
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list((v for v in m.uncertain_param_vars.values() if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))))
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        variables_in_constr = list((v for v in m.uncertain_params if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))))
        self.assertEqual(len(variables_in_constr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the AxisAlignedEllipsoidalSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        _set = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        config = Block()
        config.uncertainty_set = _set
        AxisAlignedEllipsoidalSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for AxisAlignedEllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for AxisAlignedEllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for AxisAlignedEllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for AxisAlignedEllipsoidalSet')

    def test_set_with_zero_half_lengths(self):
        half_lengths = [1, 0, 2, 0]
        center = [1, 1, 1, 1]
        ell = AxisAlignedEllipsoidalSet(center, half_lengths)
        m = ConcreteModel()
        m.v1 = Var()
        m.v2 = Var([1, 2])
        m.v3 = Var()
        conlist = ell.set_as_constraint([m.v1, m.v2, m.v3])
        eq_cons = [con for con in conlist.values() if con.equality]
        self.assertEqual(len(conlist), 3, msg=f'Constraint list for this `AxisAlignedEllipsoidalSet` should be of length 3, but is of length {len(conlist)}')
        self.assertEqual(len(eq_cons), 2, msg=f'Number of equality constraints for this`AxisAlignedEllipsoidalSet` should be 2, there are {len(eq_cons)} such constraints')

    @unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
    def test_two_stg_mod_with_axis_aligned_set(self):
        """
        Test two-stage model with `AxisAlignedEllipsoidalSet`
        as the uncertainty set.
        """
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)
        m.con1 = Constraint(expr=m.x1 * m.u1 ** 0.5 - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u1 == m.x3)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)
        ellipsoid = AxisAlignedEllipsoidalSet(center=[1.125, 1], half_lengths=[1, 0])
        pyros_solver = SolverFactory('pyros')
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory('baron')
        results = pyros_solver.solve(model=m, first_stage_variables=[m.x1, m.x2], second_stage_variables=[], uncertain_params=[m.u1, m.u2], uncertainty_set=ellipsoid, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True})
        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Did not identify robust optimal solution to problem instance.')
        self.assertGreater(results.iterations, 0, msg='Robust infeasible model terminated in 0 iterations (nominal case).')