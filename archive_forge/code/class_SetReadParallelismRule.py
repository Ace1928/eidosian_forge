import math
from typing import Optional, Tuple, Union
from ray import available_resources as ray_available_resources
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _autodetect_parallelism
from ray.data.context import WARN_PREFIX, DataContext
from ray.data.datasource.datasource import Datasource, Reader
class SetReadParallelismRule(Rule):
    """
    This rule sets the read op's task parallelism based on the target block
    size, the requested parallelism, the number of read files, and the
    available resources in the cluster.

    If the parallelism is lower than requested, this rule also sets a split
    factor to split the output blocks of the read task, so that the following
    stage will have the desired parallelism.
    """

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        ops = [plan.dag]
        while len(ops) > 0:
            op = ops.pop(0)
            if isinstance(op, InputDataBuffer):
                continue
            logical_op = plan.op_map[op]
            if isinstance(logical_op, Read):
                self._apply(op, logical_op)
            ops += op.input_dependencies
        return plan

    def _apply(self, op: PhysicalOperator, logical_op: Read):
        detected_parallelism, reason, estimated_num_blocks, k = compute_additional_split_factor(logical_op._datasource_or_legacy_reader, logical_op._parallelism, logical_op._mem_size, op.actual_target_max_block_size, op._additional_split_factor)
        if logical_op._parallelism == -1:
            assert reason != ''
            logger.get_logger().info(f'Using autodetected parallelism={detected_parallelism} for stage {logical_op.name} to satisfy {reason}.')
        logical_op.set_detected_parallelism(detected_parallelism)
        if k is not None:
            logger.get_logger().info(f'To satisfy the requested parallelism of {detected_parallelism}, each read task output is split into {k} smaller blocks.')
        if k is not None:
            op.set_additional_split_factor(k)
        logger.get_logger().debug(f'Estimated num output blocks {estimated_num_blocks}')