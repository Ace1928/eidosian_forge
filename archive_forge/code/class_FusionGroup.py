from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
@dataclass
class FusionGroup:
    top_node_idx: int
    nodes: NodeSet
    inputs: NodeSet
    nodes_need_process: NodeSet

    def add_node(self, node):
        """
            Add a node to fusion group.
            """
        if node in self.nodes:
            return
        self.nodes_need_process.add(node)
        self.nodes.add(node)
        self.inputs.discard(node)
        self.inputs.update({n for n in node.all_input_nodes if n.op in CALLABLE_NODE_OPS and n not in self.nodes})