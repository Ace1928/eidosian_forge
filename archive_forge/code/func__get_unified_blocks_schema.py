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
def _get_unified_blocks_schema(self, blocks: BlockList, fetch_if_missing: bool=False) -> Union[type, 'pyarrow.lib.Schema']:
    """Get the unified schema of the blocks.

        Args:
            blocks: the blocks to get schema
            fetch_if_missing: Whether to execute the blocks to fetch the schema.
        """
    if isinstance(blocks, LazyBlockList):
        blocks.ensure_metadata_for_first_block()
    metadata = blocks.get_metadata(fetch_if_missing=False)
    unified_schema = unify_block_metadata_schema(metadata)
    if unified_schema is not None:
        return unified_schema
    if not fetch_if_missing:
        return None
    for _, m in blocks.iter_blocks_with_metadata():
        if m.schema is not None and (m.num_rows is None or m.num_rows > 0):
            return m.schema
    return None