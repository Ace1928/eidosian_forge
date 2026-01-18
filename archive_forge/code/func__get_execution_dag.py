from typing import Any, Iterator, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ActorPoolStrategy, get_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.interfaces.logical_plan import LogicalPlan
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.logical.util import record_operators_usage
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.plan import AllToAllStage, ExecutionPlan, OneToOneStage, Stage
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stage_impl import LimitStage, RandomizeBlocksStage
from ray.data._internal.stats import DatasetStats, StatsDict
from ray.data.block import Block, BlockMetadata, CallableClass, List
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _get_execution_dag(executor: Executor, plan: ExecutionPlan, allow_clear_input_blocks: bool, preserve_order: bool) -> Tuple[PhysicalOperator, DatasetStats]:
    """Get the physical operators DAG from a plan."""
    if hasattr(plan, '_logical_plan') and plan._logical_plan is not None:
        record_operators_usage(plan._logical_plan.dag)
    if DataContext.get_current().optimizer_enabled:
        dag = get_execution_plan(plan._logical_plan).dag
        stats = _get_initial_stats_from_plan(plan)
    else:
        dag, stats = _to_operator_dag(plan, allow_clear_input_blocks)
    if preserve_order or plan.require_preserve_order():
        executor._options.preserve_order = True
    return (dag, stats)