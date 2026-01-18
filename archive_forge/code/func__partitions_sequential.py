import itertools
from typing import Any, List, OrderedDict, Set, Optional, Callable
import operator
from torch.fx import Node
import torch
from torch.fx.passes.utils.source_matcher_utils import (
def _partitions_sequential(partitions: List[SourcePartition]):
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and (not check_subgraphs_connected(prev_partition, partition)):
            return False
        prev_partition = partition
    return True