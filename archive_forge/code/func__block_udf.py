import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _block_udf(block: 'pyarrow.Table') -> 'pyarrow.Table':
    from ray.data.extensions import ArrowTensorArray
    for tensor_col_name, (dtype, shape) in tensor_column_schema.items():
        np_col = _create_possibly_ragged_ndarray([np.ndarray(shape, buffer=buf.as_buffer(), dtype=dtype) for buf in block.column(tensor_col_name)])
        block = block.set_column(block._ensure_integer_index(tensor_col_name), tensor_col_name, ArrowTensorArray.from_numpy(np_col))
    if existing_block_udf is not None:
        block = existing_block_udf(block)
    return block