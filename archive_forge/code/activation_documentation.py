import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm._C import ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.utils import set_weight_attrs
PyTorch-native implementation equivalent to forward().