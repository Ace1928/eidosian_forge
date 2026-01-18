import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _maybe_insert_input_equalization_observers_for_node(node: Node, equalization_qconfig: Any, model: torch.nn.Module, named_modules: Dict[str, torch.nn.Module], graph: Graph, is_branch: bool) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(node, named_modules):
        return
    if is_branch:
        warnings.warn(f'Cannot equalize {node} because it is part of a branch.')
        return
    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue
        is_weight = node_arg_is_weight(node, arg)
        act_eq_process_ctr = equalization_qconfig.weight if is_weight else equalization_qconfig.input_activation
        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = _insert_obs_or_fq(arg, new_eq_obs_mod, model, named_modules, graph)
        new_args.append(new_eq_obs_node)
    node.args = tuple(new_args)