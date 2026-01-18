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
def _can_fuse(self, down_op: PhysicalOperator, up_op: PhysicalOperator) -> bool:
    """Returns whether the provided downstream operator can be fused with the given
        upstream operator.

        We currently support fusing two operators if the following are all true:
            * We are fusing either MapOperator -> MapOperator or
              MapOperator -> AllToAllOperator.
            * They either use the same compute configuration, or the upstream operator
              uses a task pool while the downstream operator uses an actor pool.
            * If both operators involve callable classes, the callable classes are
              the same class AND constructor args are the same for both.
            * They have compatible remote arguments.
        """
    from ray.data._internal.logical.operators.map_operator import AbstractMap, AbstractUDFMap
    if not (isinstance(up_op, TaskPoolMapOperator) and isinstance(down_op, (TaskPoolMapOperator, ActorPoolMapOperator)) or (isinstance(up_op, TaskPoolMapOperator) and isinstance(down_op, AllToAllOperator))):
        return False
    down_logical_op = self._op_map[down_op]
    up_logical_op = self._op_map[up_op]
    if up_op.get_additional_split_factor() > 1:
        return False
    if not down_logical_op._input_dependencies:
        return False
    if not (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, AbstractMap) or (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, RandomShuffle)) or (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, Repartition))):
        return False
    if isinstance(down_logical_op, Repartition) and (not down_logical_op._shuffle):
        return False
    if isinstance(down_logical_op, AbstractUDFMap) and isinstance(up_logical_op, AbstractUDFMap):
        if is_task_compute(down_logical_op._compute) and get_compute(up_logical_op._compute) != get_compute(down_logical_op._compute):
            return False
    if not _are_remote_args_compatible(getattr(up_logical_op, '_ray_remote_args', {}), getattr(down_logical_op, '_ray_remote_args', {})):
        return False
    if not self._can_merge_target_max_block_size(up_op.target_max_block_size, down_op.target_max_block_size):
        return False
    return True