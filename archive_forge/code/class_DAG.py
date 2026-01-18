import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
class DAG:
    """DAG class contains all the DAG nodes"""

    def __init__(self) -> None:
        self.nodes: List[DAGNode] = []

    def create_node(self, submodule_node: Node, input_nodes: List[Node], output_nodes: List[Node], logical_devices: List[int], size_bytes: int) -> None:
        node = DAGNode(submodule_node, input_nodes, output_nodes, logical_devices, size_bytes)
        self.nodes.append(node)