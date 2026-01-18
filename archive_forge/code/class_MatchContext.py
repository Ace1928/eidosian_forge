from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
class MatchContext:
    """
    State needed while running PatternExpr._match().
    """

    def __init__(self, outputs: List[Optional[PatternExpr]], pattern_to_node: Optional[Dict[PatternExpr, Node]]=None, *, graph: torch.fx.Graph):
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else pattern_to_node
        self.graph = graph
        self.exclusive_node_set: List[NodeOrConstant] = []

    def match(self, pattern, node):
        """wrapper to check reused nodes in patterns"""
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(pattern)
            else:
                return FailedMatch('repeated pattern differs')
        m = pattern._match(node, self)
        assert pattern not in self.pattern_to_node
        self.pattern_to_node[pattern] = node if m else None
        m.ctx = self
        return m

    def filter_multi_user_patterns(self):
        return {pattern: node for pattern, node in self.pattern_to_node.items() if pattern.has_multiple_users() and node is not None}