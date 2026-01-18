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
@dataclass
class _MinimizerSettingBase:
    """
    Args:
    `accumulate_error`: Instead of using a's input for both converted module to verify
    , use the previous outputs of each converted module as input to accumulate the
    errors.

    `traverse_method`: "sequential" or "binary" or "accumulate"
    Determine the way of traverse the nodes in FX module.

    `find_all`: Minimizer will go through the entire model and return all problematic nodes.

    `return_intermediate`: If true, when using `run_nodes()` function to run the
    model, intermediate results of all the ops will be returned as output.
    """
    accumulate_error: bool = False
    traverse_method: str = 'sequential'
    find_all: bool = False
    return_intermediate: bool = False

    def __str__(self):
        settings_str = 'FX Minimizer Settings:\n'
        for k, v in vars(self).items():
            settings_str += f'\t{k}: {v}\n'
        return settings_str