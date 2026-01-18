import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
def _get_all_later_node_usages(tensor_aliases: Set[Node], op_index: int):

    def _add_if_tensor(x, set_):
        if isinstance(x, FakeTensor):
            set_.add(StorageWeakRef(x._typed_storage()))
    nodes_used_after = set()
    for t in tensor_aliases:
        usage_nodes = t.users
        for n in usage_nodes:
            if 'node_idx' not in n.meta or n.meta['node_idx'] <= op_index:
                continue
            if n in tensor_aliases:
                if isinstance(n.target, torch._ops.OpOverload) or n.target == _operator.getitem:
                    continue
            nodes_used_after.add(n)
    return nodes_used_after