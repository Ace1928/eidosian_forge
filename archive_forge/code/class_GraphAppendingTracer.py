import enum
import dis
import copy
import sys
import torch
import inspect
import operator
import traceback
import collections
from dataclasses import is_dataclass, fields
from .graph import magic_methods, reflectable_magic_methods, Graph
from typing import Tuple, Dict, OrderedDict, Optional, Any, Iterator, Callable
from .node import Target, Node, Argument, base_types, map_aggregate
from ._compatibility import compatibility
from .operator_schemas import check_for_mutable_operation
import torch.fx.traceback as fx_traceback
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):

    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.scope = Scope('', None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}