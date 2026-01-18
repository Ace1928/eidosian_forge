import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(self, module: torch.fx.GraphModule, operator_support: OperatorSupportBase, allow_non_tensor: bool):
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor

    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList):
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)
            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    if not is_node_output_tensor(user):
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self):
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            non_tensor_cpu_nodes.append(node)
        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self):
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []
            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        break
            if not new_cpu_nodes:
                break
            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)
            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        submodules = dict(self.module.named_modules())
        self.acc_nodes = {n for n in self.module.graph.nodes if n.op in CALLABLE_NODE_OPS and self.operator_support.is_node_supported(submodules, n)}
        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()
        return self.acc_nodes