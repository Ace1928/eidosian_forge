import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
def _is_observed_standalone_module(module: Any) -> bool:
    return _is_observed_module(module) and module.meta['_observed_graph_module_attrs'].is_observed_standalone_module