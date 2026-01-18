import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _store_outputs(self, a_result: TensorOrTensors, b_result: TensorOrTensors, submodule: torch.fx.GraphModule):
    """
        Store the outputs of self.run_a() and self.run_b() into self.a_outputs and
        self.b_outputs, so that we can use them when execute preceding nodes that
        use those outputs as inputs.

        Args:
            a_result: Output of self.run_a(). Could be a tensor or tensors.
            b_result: Output of self.run_b(). Could be a tensor or tensors.
            submodule: The module that generates a_result and b_result.
        """
    output_node = next((node for node in submodule.graph.nodes if node.op == 'output'))
    if isinstance(output_node.args[0], torch.fx.Node):
        self.a_outputs[output_node.args[0].name] = a_result
        self.b_outputs[output_node.args[0].name] = b_result
    else:
        for i, arg in enumerate(output_node.args[0]):
            self.a_outputs[arg.name] = a_result[i]
            self.b_outputs[arg.name] = b_result[i]