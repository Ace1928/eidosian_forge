from typing import List
from ray.data._internal.logical.interfaces import (
from ray.data._internal.logical.rules._user_provided_optimizer_rules import (
from ray.data._internal.logical.rules.inherit_target_max_block_size import (
from ray.data._internal.logical.rules.operator_fusion import OperatorFusionRule
from ray.data._internal.logical.rules.randomize_blocks import ReorderRandomizeBlocksRule
from ray.data._internal.logical.rules.set_read_parallelism import SetReadParallelismRule
from ray.data._internal.logical.rules.zero_copy_map_fusion import (
from ray.data._internal.planner.planner import Planner
class PhysicalOptimizer(Optimizer):
    """The optimizer for physical operators."""

    @property
    def rules(self) -> List['Rule']:
        rules = add_user_provided_physical_rules(DEFAULT_PHYSICAL_RULES)
        return [rule_cls() for rule_cls in rules]