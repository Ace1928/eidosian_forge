import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _split_single_block(b: Block, left_size: int) -> (Block, Block):
    acc = BlockAccessor.for_block(b)
    left = acc.slice(0, left_size)
    right = acc.slice(left_size, acc.num_rows())
    assert BlockAccessor.for_block(left).num_rows() == left_size
    assert BlockAccessor.for_block(right).num_rows() == acc.num_rows() - left_size
    return (left, right)