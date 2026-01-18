import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def _outline_submodules(orig_graph: torch.fx.Graph, root_module: torch.fx.GraphModule):
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    ModuleFrame(orig_graph, seen_nodes, seen_modules, None, [''], '', {entry.fqn: entry.signature for entry in root_module.module_call_graph if entry.signature}, graph_module=root_module).run_outer()