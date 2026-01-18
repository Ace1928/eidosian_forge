import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def _reorder_stages(stages: List[Stage]) -> List[Stage]:
    """Reorder randomize stages to the end to enable better stage fusion.

    This applies to RandomizeBlockOrder stages specifically (issue #26057).

    Args:
        stages: Stages to try to reorder.

    Returns:
        Reordered stages.
    """
    from ray.data._internal.stage_impl import RandomizeBlocksStage
    output: List[Stage] = []
    reorder_buf: List[RandomizeBlocksStage] = []
    for s in stages:
        if isinstance(s, RandomizeBlocksStage):
            reorder_buf.append(s)
        else:
            if isinstance(s, AllToAllStage) or s.name == 'Write':
                output.extend(reorder_buf)
                reorder_buf = []
            output.append(s)
    output.extend(reorder_buf)
    return output