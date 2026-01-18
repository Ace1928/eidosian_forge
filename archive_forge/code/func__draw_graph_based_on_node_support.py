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
def _draw_graph_based_on_node_support(self, mod: torch.fx.GraphModule, supported_nodes: NodeList):
    color_map = {'default': 'AliceBlue', 'supported': 'chartreuse1', 'unsupported': 'crimson'}

    class CustomDrawer(FxGraphDrawer):

        def _get_node_style(self, node):
            template = super()._get_node_style(node)
            if node in supported_nodes:
                template['fillcolor'] = color_map['supported']
            elif node.op in CALLABLE_NODE_OPS:
                template['fillcolor'] = color_map['unsupported']
            else:
                template['fillcolor'] = color_map['default']
            return template
    drawer = CustomDrawer(mod, 'node_support', ignore_getattr=True)
    dot_graph = drawer.get_main_dot_graph()
    dot_graph.write_raw('node_support.dot')