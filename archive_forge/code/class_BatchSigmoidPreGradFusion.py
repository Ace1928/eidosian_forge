import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
@register_fusion('batch_sigmoid')
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):

    def __init__(self, **kwargs):
        super().__init__(torch.sigmoid, **kwargs)