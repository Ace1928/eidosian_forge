from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
class IndexedConstraintsInDisj(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)

    def test_transformed_constraints_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        transBlock = m.component('_pyomo_gdp_bigm_reformulation')
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.component('relaxedDisjuncts')
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        cons11 = bigm.get_transformed_constraints(m.disjunct[0].c[1])
        self.assertEqual(len(cons11), 1)
        cons11_lb = cons11[0]
        self.assertIsInstance(cons11_lb.parent_component(), Constraint)
        self.assertTrue(cons11_lb.active)
        cons12 = bigm.get_transformed_constraints(m.disjunct[0].c[2])
        self.assertEqual(len(cons12), 1)
        cons12_lb = cons12[0]
        self.assertIsInstance(cons12_lb.parent_component(), Constraint)
        self.assertTrue(cons12_lb.active)
        cons21 = bigm.get_transformed_constraints(m.disjunct[1].c[1])
        self.assertEqual(len(cons21), 2)
        cons21_lb = cons21[0]
        cons21_ub = cons21[1]
        self.assertIsInstance(cons21_lb.parent_component(), Constraint)
        self.assertIsInstance(cons21_ub.parent_component(), Constraint)
        self.assertTrue(cons21_lb.active)
        self.assertTrue(cons21_ub.active)
        cons22 = bigm.get_transformed_constraints(m.disjunct[1].c[2])
        self.assertEqual(len(cons22), 2)
        cons22_lb = cons22[0]
        cons22_ub = cons22[1]
        self.assertIsInstance(cons22_lb.parent_component(), Constraint)
        self.assertIsInstance(cons22_ub.parent_component(), Constraint)
        self.assertTrue(cons22_lb.active)
        self.assertTrue(cons22_ub.active)

    def checkMs(self, model, c11lb, c12lb, c21lb, c21ub, c22lb, c22ub):
        bigm = TransformationFactory('gdp.bigm')
        c = bigm.get_transformed_constraints(model.disjunct[0].c[1])
        self.assertEqual(len(c), 1)
        lb = c[0]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c11lb)
        ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c11lb)
        c = bigm.get_transformed_constraints(model.disjunct[0].c[2])
        self.assertEqual(len(c), 1)
        lb = c[0]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c12lb)
        ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c12lb)
        c = bigm.get_transformed_constraints(model.disjunct[1].c[1])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21lb)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21ub)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21ub)
        c = bigm.get_transformed_constraints(model.disjunct[1].c[2])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22lb)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22ub)
        ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22ub)

    def test_arg_M_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c[1]] = 19
        TransformationFactory('gdp.bigm').apply_to(m, bigM={None: 19, m.disjunct[0].c[1]: 17, m.disjunct[0].c[2]: 18})
        self.checkMs(m, -17, -18, -19, 19, -19, 19)

    def test_arg_M_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m, bigM=ComponentMap({None: 19, m.disjunct[0].c: 17}))
        self.checkMs(m, -17, -17, -19, 19, -19, 19)

    def test_suffix_M_None_on_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_None_on_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c[1]] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -20, -20, 20, -20, 20)

    def test_suffix_M_indexedConstraint_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_constraintData_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19
        m.BigM[m.disjunct[0].c[1]] = 18
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -19, -20, 20, -20, 20)

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        self.diff_apply_to_and_create_using(m)