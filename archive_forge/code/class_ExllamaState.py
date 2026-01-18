import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from fractions import Fraction
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.quantization.base_config import (
class ExllamaState(Enum):
    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()