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
def _add_matched_node_name_to_set(matched_node_pattern: NodePattern, s: Set[str]):
    if isinstance(matched_node_pattern, Node):
        s.add(matched_node_pattern.name)
    elif isinstance(matched_node_pattern, (list, tuple)):
        for maybe_node in matched_node_pattern:
            _add_matched_node_name_to_set(maybe_node, s)