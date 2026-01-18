import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def _consolidate_placeholders(cur_graph, inps):
    new_graph = fx.Graph()
    env = {}
    seen_non_placeholder = False
    for node in cur_graph.nodes:
        if node.op == 'placeholder':
            new_node = new_graph.node_copy(node, lambda x: env[x])
            env[node] = new_node
        elif not seen_non_placeholder and is_load_tensor_node(node):
            new_node = new_graph.placeholder(node.name)
            env[node] = new_node
            inps.append(torch.ops.debugprims.load_tensor.default(*node.args, **node.kwargs))
        else:
            seen_non_placeholder = True
    for node in cur_graph.nodes:
        if node not in env:
            new_node = new_graph.node_copy(node, lambda x: env[x])
            env[node] = new_node
    return new_graph