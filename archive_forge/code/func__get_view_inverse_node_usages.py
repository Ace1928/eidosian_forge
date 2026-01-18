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
def _get_view_inverse_node_usages(later_node_usages: Set[Node], self_aliases: Set[Node]) -> Set[Node]:

    def matching_view_metadata(a, b):
        return a.size() == b.size() and a.stride() == b.stride() and (a.storage_offset() == b.storage_offset())
    view_inverse_nodes = set()
    for n in sorted(later_node_usages, key=lambda x: x.meta['node_idx']):
        if n.target not in _VIEW_INVERSE_MAP:
            continue
        base = n.args[0]
        mutated_view = n.args[1]
        assert isinstance(base, Node)
        assert isinstance(base.meta['fake_result'], FakeTensor)
        assert isinstance(mutated_view, Node)
        assert isinstance(mutated_view.meta['fake_result'], FakeTensor)
        original_view = _VIEW_INVERSE_MAP[n.target]
        for self_alias in self_aliases:
            if 'view_of' not in self_alias.meta:
                continue
            self_alias_base = self_alias.meta['view_of']
            try:
                view_replay_metadata = original_view(self_alias_base.meta['fake_result'], *n.args[2:], **n.kwargs)
                expected_metadata = self_alias.meta['fake_result']
                if matching_view_metadata(self_alias_base.meta['fake_result'], base.meta['fake_result']) and matching_view_metadata(view_replay_metadata, expected_metadata):
                    view_inverse_nodes.add(n)
            except Exception:
                continue
    return view_inverse_nodes