import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def _get_logger_for_subgraph(model: GraphModule, first_node: Node, last_node: Node, subgraph_idx: int, subgraph_candidate_idx: int, qconfig_str: str, logger_cls: Callable, fqn: Optional[str]) -> torch.nn.Module:
    """
    Given a model and a linear subgraph starting from `first_node` and
    ending with `last_node`, creates a logger for the end of this
    subgraph.
    """
    if fqn is None:
        fqn = ''
    logger_mod_orig = logger_cls(first_node.name, last_node.name, f'subgraph_{subgraph_idx}_{subgraph_candidate_idx}', 'model', get_target_type_str(last_node, model), get_target_type_str(first_node, model), NSSingleResultValuesType.NODE_OUTPUT.value, 0, 0, fqn, qconfig_str)
    logger_mod_orig.enabled = False
    return logger_mod_orig