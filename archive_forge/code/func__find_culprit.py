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
def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors) -> str:
    """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """
    return 'Unable to find a culprit because _find_culprit() function is not implemented.'