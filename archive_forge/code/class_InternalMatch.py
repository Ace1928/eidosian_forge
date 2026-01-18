from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch:
    anchors: List[Node]
    nodes_map: Dict[Node, Node] = field(default_factory=dict)
    placeholder_nodes: List[Node] = field(default_factory=list)
    returning_nodes: List[Node] = field(default_factory=list)
    name_node_map: Dict[str, Node] = field(default_factory=dict)

    def __copy__(self):
        return InternalMatch(anchors=self.anchors, nodes_map=self.nodes_map.copy(), placeholder_nodes=self.placeholder_nodes.copy(), returning_nodes=self.returning_nodes.copy())