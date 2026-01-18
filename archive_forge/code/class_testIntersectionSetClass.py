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
class testIntersectionSetClass(unittest.TestCase):
    """
    Unit tests for the IntersectionSet class.
    Required input is set objects to intersect,
    and set_as_constraint requires
    an NLP solver to confirm the intersection is not empty.
    """

    def test_normal_construction_and_update(self):
        """
        Test IntersectionSet constructor and setter
        work normally when arguments are appropriate.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
        iset = IntersectionSet(box_set=bset, axis_aligned_set=aset)
        self.assertIn(bset, iset.all_sets, msg="IntersectionSet 'all_sets' attribute does notcontain expected BoxSet")
        self.assertIn(aset, iset.all_sets, msg="IntersectionSet 'all_sets' attribute does notcontain expected AxisAlignedEllipsoidalSet")

    def test_error_on_intersecting_wrong_dims(self):
        """
        Test ValueError raised if IntersectionSet sets
        are not of same dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
        wrong_aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
        exc_str = '.*of dimension 2, but attempting to add set of dimension 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, wrong_set=wrong_aset)
        iset = IntersectionSet(box_set=bset, axis_set=aset)
        with self.assertRaisesRegex(ValueError, exc_str):
            iset.all_sets.append(wrong_aset)

    def test_type_error_on_invalid_arg(self):
        """
        Test TypeError raised if an argument not of type
        UncertaintySet is passed to the IntersectionSet
        constructor or appended to 'all_sets'.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
        exc_str = "Entry '1' of the argument `all_sets` is not An `UncertaintySet` object.*\\(provided type 'int'\\)"
        with self.assertRaisesRegex(TypeError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, invalid_arg=1)
        iset = IntersectionSet(box_set=bset, axis_set=aset)
        with self.assertRaisesRegex(TypeError, exc_str):
            iset.all_sets.append(1)

    def test_error_on_intersection_dim_change(self):
        """
        IntersectionSet dimension is considered immutable.
        Test ValueError raised when attempting to set the
        constituent sets to a different dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
        iset = IntersectionSet(box_set=bset, axis_set=aset)
        exc_str = 'Attempting to set.*dimension 2 to a sequence.* of dimension 1'
        with self.assertRaisesRegex(ValueError, exc_str):
            iset.all_sets = [BoxSet([[1, 1]]), AxisAlignedEllipsoidalSet([0], [1])]

    def test_error_on_too_few_sets(self):
        """
        Check ValueError raised if too few sets are passed
        to the intersection set.
        """
        exc_str = 'Attempting.*minimum required length 2.*iterable of length 1'
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(bset=BoxSet([[1, 2]]))
        iset = IntersectionSet(box_set=BoxSet([[1, 2]]), axis_set=AxisAlignedEllipsoidalSet([0], [1]))
        with self.assertRaisesRegex(ValueError, exc_str):
            iset.all_sets = [BoxSet([[1, 1]])]

    def test_intersection_uncertainty_set_list_behavior(self):
        """
        Test the 'all_sets' attribute of the IntersectionSet
        class behaves like a regular Python list.
        """
        iset = IntersectionSet(bset=BoxSet([[0, 2]]), aset=AxisAlignedEllipsoidalSet([0], [1]))
        all_sets = iset.all_sets
        all_sets.append(BoxSet([[1, 2]]))
        del all_sets[2:]
        all_sets.extend([BoxSet([[1, 2]]), EllipsoidalSet([0], [[1]], 2)])
        del all_sets[2:]
        all_sets[0]
        all_sets[1]
        all_sets[100:]
        all_sets[0:2:20]
        all_sets[0:2:1]
        all_sets[-20:-1:2]
        self.assertRaises(IndexError, lambda: all_sets[2])
        self.assertRaises(IndexError, lambda: all_sets[-3])
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            all_sets[:] = all_sets[0]
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            del all_sets[1]
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            del all_sets[1:]
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            del all_sets[:]
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            all_sets.clear()
        with self.assertRaisesRegex(ValueError, 'Length.* must be at least 2'):
            all_sets[0:] = []
        with self.assertRaisesRegex(IndexError, 'assignment index out of range'):
            all_sets[-3] = BoxSet([[1, 1.5]])
        with self.assertRaisesRegex(IndexError, 'assignment index out of range'):
            all_sets[2] = BoxSet([[1, 1.5]])
        all_sets[3:] = [BoxSet([[1, 1.5]]), BoxSet([[1, 3]])]

    @unittest.skipUnless(ipopt_available, 'IPOPT is not available.')
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
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        config = ConfigBlock()
        solver = SolverFactory('ipopt')
        config.declare('global_solver', ConfigValue(default=solver))
        m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list((id(u) for u in uncertain_params_in_expr)):
                        uncertain_params_in_expr.append(v)
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    @unittest.skipUnless(ipopt_available, 'IPOPT is not available.')
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
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        solver = SolverFactory('ipopt')
        config = ConfigBlock()
        config.declare('global_solver', ConfigValue(default=solver))
        m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
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
        Q1 = BoxSet(bounds=bounds)
        Q2 = BoxSet(bounds=[(-2, 1), (-1, 2)])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        self.assertTrue(Q.point_in_set([0, 0]), msg='Point is not in the IntersectionSet.')

    @unittest.skipUnless(baron_available, 'Global NLP solver is not available.')
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        bounds = [(-1, 1), (-1, 1)]
        Q1 = BoxSet(bounds=bounds)
        Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[5, 5])
        Q = IntersectionSet(Q1=Q1, Q2=Q2)
        config = Block()
        config.uncertainty_set = Q
        config.global_solver = SolverFactory('baron')
        IntersectionSet.add_bounds_on_uncertain_parameters(m, config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for IntersectionSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for IntersectionSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for IntersectionSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for IntersectionSet')