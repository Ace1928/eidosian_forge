import math
from typing import Callable, Iterable, List, Optional, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
class PushBasedShufflePartitionOp(_ShufflePartitionOp, PushBasedShufflePlan):
    pass