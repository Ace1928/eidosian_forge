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
@compatibility(is_backward_compatible=False)
class FxNetMinimizerBadModuleError(Exception):
    """
    Raised if failed to split out a minimize module
    """
    pass