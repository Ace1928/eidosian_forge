from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
class MultiTermDisj(unittest.TestCase, CommonTests):

    def test_xor_constraint(self):
        ct.check_three_term_xor_constraint(self, 'hull')

    def test_create_using(self):
        m = models.makeThreeTermIndexedDisj()
        self.diff_apply_to_and_create_using(m)

    def test_do_not_disaggregate_more_than_necessary(self):
        m = models.makeThreeTermDisjunctionWithOneVarInOneDisjunct()
        hull = TransformationFactory('gdp.hull')
        hull.apply_to(m)
        x1 = hull.get_disaggregated_var(m.x, m.d1)
        self.assertEqual(x1.lb, -2)
        self.assertEqual(x1.ub, 8)
        self.assertIs(hull.get_src_var(x1), m.x)
        x2 = m.disjunction.algebraic_constraint.parent_block()._disaggregatedVars[0]
        self.assertIs(hull.get_src_var(x2), m.x)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d2), x2)
        self.assertIs(hull.get_disaggregated_var(m.x, m.d3), x2)
        bounds = hull.get_var_bounds_constraint(x2)
        self.assertEqual(len(bounds), 2)
        self.assertIsNone(bounds['lb'].lower)
        self.assertEqual(bounds['lb'].upper, 0)
        repn = generate_standard_repn(bounds['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertIs(repn.linear_vars[0], m.d1.indicator_var.get_associated_binary())
        self.assertEqual(repn.linear_coefs[0], 2)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.constant, -2)
        self.assertIsNone(bounds['ub'].lower)
        self.assertEqual(bounds['ub'].upper, 0)
        repn = generate_standard_repn(bounds['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], x2)
        self.assertIs(repn.linear_vars[1], m.d1.indicator_var.get_associated_binary())
        self.assertEqual(repn.linear_coefs[1], 8)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.constant, -8)
        c = hull.get_disaggregation_constraint(m.x, m.disjunction)
        self.assertEqual(c.lower, 0)
        self.assertEqual(c.upper, 0)
        repn = generate_standard_repn(c.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        self.assertIs(repn.linear_vars[0], m.x)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertIs(repn.linear_vars[2], x1)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], -1)
        self.assertEqual(repn.linear_coefs[2], -1)
        self.assertEqual(repn.constant, 0)