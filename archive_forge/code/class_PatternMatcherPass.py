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
class PatternMatcherPass:

    def __init__(self, prevent_match_across_mutations=False):
        super().__init__()
        self.patterns: DefaultDict[torch.fx.node.Target, List[PatternEntry]] = defaultdict(list)
        self.prevent_match_across_mutations = prevent_match_across_mutations

    def __getitem__(self, item: torch.fx.node.Target) -> List[PatternEntry]:
        return self.patterns[item]

    def apply(self, graph: torch.fx.GraphModule) -> int:
        if not self.patterns:
            return 0
        if isinstance(graph, torch.fx.GraphModule):
            graph = graph.graph
        if self.prevent_match_across_mutations:
            if should_compute_mutation_region_ids(graph):
                compute_mutation_region_ids(graph)
            get_mutation_region_id_partial = functools.partial(get_mutation_region_id, graph)
        count = 0
        for node in reversed(graph.nodes):
            target = extract_target(node)
            if node.op in ['call_function', 'call_method', 'call_module'] and target in self.patterns:
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue
                for entry in self.patterns[target]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    if self.prevent_match_across_mutations and is_match(m) and (len(set(map(get_mutation_region_id_partial, m.nodes))) != 1):
                        continue
                    if os.environ.get('TORCHINDUCTOR_PATTERN_MATCH_DEBUG') == node.name:
                        log.warning('%s%s %s %s', node, node.args, m, entry.pattern)
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)
                        counters['inductor']['pattern_matcher_count'] += 1
                        counters['inductor']['pattern_matcher_nodes'] += len(m.nodes)
        return count

    def clear(self):
        self.patterns.clear()