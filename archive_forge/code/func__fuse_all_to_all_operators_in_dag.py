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
def _fuse_all_to_all_operators_in_dag(self, dag: AllToAllOperator) -> AllToAllOperator:
    """Starting at the given operator, traverses up the DAG of operators
        and recursively fuses compatible MapOperator -> AllToAllOperator pairs.

        Also, sets the target block size of the immediately upstream map op to
        match the shuffle block size. We use a larger block size for shuffles
        because tiny blocks are bad for I/O performance.

        Returns the current (root) operator after completing upstream operator fusions.
        """
    upstream_ops = dag.input_dependencies
    while len(upstream_ops) == 1 and isinstance(dag, AllToAllOperator) and isinstance(upstream_ops[0], MapOperator) and self._can_fuse(dag, upstream_ops[0]):
        dag = self._get_fused_all_to_all_operator(dag, upstream_ops[0])
        upstream_ops = dag.input_dependencies
    dag._input_dependencies = [self._fuse_all_to_all_operators_in_dag(upstream_op) for upstream_op in upstream_ops]
    return dag