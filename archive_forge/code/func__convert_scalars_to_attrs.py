import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization.quantizer.utils import (
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in model.graph.nodes:
        if n.op != 'call_function' or n.target not in [torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = '_tensor_constant_'
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            model.register_buffer(tensor_constant_name, torch.tensor(float(args[i])))
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node('get_attr', tensor_constant_name, (), {})
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model