import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
class CommonModels(unittest.TestCase, CommonTests):

    def test_user_deactivated_disjuncts(self):
        ct.check_user_deactivated_disjuncts(self, 'partition_disjuncts', check_trans_block=False, num_partitions=2)

    def test_improperly_deactivated_disjuncts(self):
        ct.check_improperly_deactivated_disjuncts(self, 'partition_disjuncts', num_partitions=2)

    def test_do_not_transform_userDeactivated_indexedDisjunction(self):
        ct.check_do_not_transform_userDeactivated_indexedDisjunction(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunction_deactivated(self):
        ct.check_disjunction_deactivated(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunctDatas_deactivated(self):
        ct.check_disjunctDatas_deactivated(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_constraints(self):
        ct.check_deactivated_constraints(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjuncts(self):
        ct.check_deactivated_disjuncts(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjunctions(self):
        ct.check_deactivated_disjunctions(self, 'partition_disjuncts', num_partitions=2)

    def test_constraints_deactivated_indexedDisjunction(self):
        ct.check_constraints_deactivated_indexedDisjunction(self, 'partition_disjuncts', num_partitions=2)

    def test_only_targets_inactive(self):
        ct.check_only_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_target_not_a_component_error(self):
        ct.check_target_not_a_component_error(self, 'partition_disjuncts', num_partitions=2)

    def test_indexedDisj_targets_inactive(self):
        ct.check_indexedDisj_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_warn_for_untransformed(self):
        ct.check_warn_for_untransformed(self, 'partition_disjuncts', num_partitions=2)

    def test_disjData_targets_inactive(self):
        ct.check_disjData_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_indexedBlock_targets_inactive(self):
        ct.check_indexedBlock_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_blockData_targets_inactive(self):
        ct.check_blockData_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_transformation_simple_block(self):
        ct.check_transformation_simple_block(self, 'partition_disjuncts', num_partitions=2)

    def test_transform_block_data(self):
        ct.check_transform_block_data(self, 'partition_disjuncts', num_partitions=2)

    def test_simple_block_target(self):
        ct.check_simple_block_target(self, 'partition_disjuncts', num_partitions=2)

    def test_block_data_target(self):
        ct.check_block_data_target(self, 'partition_disjuncts', num_partitions=2)

    def test_indexed_block_target(self):
        ct.check_indexed_block_target(self, 'partition_disjuncts', num_partitions=2)

    def test_block_targets_inactive(self):
        ct.check_block_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_transform_empty_disjunction(self):
        ct.check_transform_empty_disjunction(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjunct_nonzero_indicator_var(self):
        ct.check_deactivated_disjunct_nonzero_indicator_var(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjunct_unfixed_indicator_var(self):
        ct.check_deactivated_disjunct_unfixed_indicator_var(self, 'partition_disjuncts', num_partitions=2)

    def test_silly_target(self):
        ct.check_silly_target(self, 'partition_disjuncts', num_partitions=2)

    def test_error_for_same_disjunct_in_multiple_disjunctions(self):
        ct.check_error_for_same_disjunct_in_multiple_disjunctions(self, 'partition_disjuncts', num_partitions=2)

    def test_cannot_call_transformation_on_disjunction(self):
        ct.check_cannot_call_transformation_on_disjunction(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunction_target_err(self):
        ct.check_disjunction_target_err(self, 'partition_disjuncts', num_partitions=2)

    def test_disjuncts_inactive_nested(self):
        ct.check_disjuncts_inactive_nested(self, 'partition_disjuncts', num_partitions=2)

    def test_deactivated_disjunct_leaves_nested_disjunct_active(self):
        ct.check_deactivated_disjunct_leaves_nested_disjunct_active(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunct_targets_inactive(self):
        ct.check_disjunct_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_disjunctData_targets_inactive(self):
        ct.check_disjunctData_targets_inactive(self, 'partition_disjuncts', num_partitions=2)

    def test_RangeSet(self):
        ct.check_RangeSet(self, 'partition_disjuncts', num_partitions=2)

    def test_Expression(self):
        ct.check_Expression(self, 'partition_disjuncts', num_partitions=2)

    def test_untransformed_network_raises_GDPError(self):
        ct.check_untransformed_network_raises_GDPError(self, 'partition_disjuncts', num_partitions=2)

    @unittest.skipUnless(ct.linear_solvers, 'Could not find a linear solver')
    def test_network_disjuncts(self):
        ct.check_network_disjuncts(self, True, 'between_steps', num_partitions=2)
        ct.check_network_disjuncts(self, False, 'between_steps', num_partitions=2)