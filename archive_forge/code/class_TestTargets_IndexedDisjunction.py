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
class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):

    def test_indexedDisj_targets_inactive(self):
        ct.check_indexedDisj_targets_inactive(self, 'hull')

    def test_indexedDisj_only_targets_transformed(self):
        ct.check_indexedDisj_only_targets_transformed(self, 'hull')

    def test_warn_for_untransformed(self):
        ct.check_warn_for_untransformed(self, 'hull')

    def test_disjData_targets_inactive(self):
        ct.check_disjData_targets_inactive(self, 'hull')
        m = models.makeDisjunctionsOnIndexedBlock()

    def test_disjData_only_targets_transformed(self):
        ct.check_disjData_only_targets_transformed(self, 'hull')

    def test_indexedBlock_targets_inactive(self):
        ct.check_indexedBlock_targets_inactive(self, 'hull')

    def test_indexedBlock_only_targets_transformed(self):
        ct.check_indexedBlock_only_targets_transformed(self, 'hull')

    def test_blockData_targets_inactive(self):
        ct.check_blockData_targets_inactive(self, 'hull')

    def test_blockData_only_targets_transformed(self):
        ct.check_blockData_only_targets_transformed(self, 'hull')

    def test_do_not_transform_deactivated_targets(self):
        ct.check_do_not_transform_deactivated_targets(self, 'hull')

    def test_create_using(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        ct.diff_apply_to_and_create_using(self, m, 'gdp.hull')