import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
class NodeInputOrOutputType(enum.Enum):
    FP32 = enum.auto()
    INT8 = enum.auto()
    FP16 = enum.auto()
    UNKNOWN = enum.auto()
    FP32_OR_INT8 = enum.auto()