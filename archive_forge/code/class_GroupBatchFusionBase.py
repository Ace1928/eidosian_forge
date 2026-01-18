import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
class GroupBatchFusionBase:

    def __init__(self, **kwargs):
        self.graph_search_options = kwargs.pop('graph_search_options', default_graph_search_options)

    def match(self, node):
        raise NotImplementedError('match called on base')

    def fuse(self, graph, subset):
        raise NotImplementedError('fuse called on base')