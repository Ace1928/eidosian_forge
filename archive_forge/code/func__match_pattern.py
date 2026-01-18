import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def _match_pattern(match_pattern: List[Callable]) -> Optional[Node]:
    """
        Traverse up the graph and match the args one by one.
        If there is a match, return the last matched node, or None otherwise.
        """
    a = arg
    for i, match in enumerate(match_pattern):
        if not match(a):
            return None
        if i < len(match_pattern) - 1:
            if match == match_tuple:
                a = a.args[0][0]
            else:
                a = a.args[0]
    return a