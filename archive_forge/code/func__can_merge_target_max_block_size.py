from typing import List, Optional, Tuple
from ray.data._internal.compute import get_compute, is_task_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.actor_pool_map_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def _can_merge_target_max_block_size(self, up_target_max_block_size: Optional[int], down_target_max_block_size: Optional[int]):
    if up_target_max_block_size is not None:
        if down_target_max_block_size is None:
            down_target_max_block_size = DataContext.get_current().target_max_block_size
        if up_target_max_block_size != down_target_max_block_size:
            return False
    return True