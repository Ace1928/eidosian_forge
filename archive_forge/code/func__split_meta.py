import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _split_meta(m: BlockMetadata, left_size: int) -> (BlockMetadata, BlockMetadata):
    left_bytes = int(math.floor(m.size_bytes * (left_size / m.num_rows)))
    left = BlockMetadata(num_rows=left_size, size_bytes=left_bytes, schema=m.schema, input_files=m.input_files, exec_stats=None)
    right = BlockMetadata(num_rows=m.num_rows - left_size, size_bytes=m.size_bytes - left_bytes, schema=m.schema, input_files=m.input_files, exec_stats=None)
    return (left, right)