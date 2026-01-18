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
def _build_submodule(self, nodes: NodeSet) -> Tuple[torch.fx.GraphModule, str]:
    """
        Split self.module so that one submodule consists of `nodes` and only `nodes`.

        Args:
            nodes: Nodes that we want to include in the minimize submodule.

        Returns:
            split_module (torch.fx.GraphModule): the module after split.
            submodule_name (str): the name of the submodule that consists of `nodes`.
        """
    self._tag_nodes(nodes)
    split_module = split_by_tags(self.module, ['main_0', 'minimize', 'main_1'])
    submodule_name: str = ''
    for child_name, _ in split_module.named_children():
        if 'minimize' not in child_name:
            continue
        if submodule_name == '':
            submodule_name = child_name
        else:
            raise FxNetMinimizerBadModuleError(f'Expected only one minimize submodule with nodes {nodes}')
    if submodule_name == '':
        raise FxNetMinimizerBadModuleError(f'Minimize submodule was not found with nodes {nodes}')
    return (split_module, submodule_name)