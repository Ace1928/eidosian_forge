import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """
    pass