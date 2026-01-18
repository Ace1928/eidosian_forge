import operator
from typing import List
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from ..fuser_method_mappings import (
from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
from .qnnpack import (

    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    