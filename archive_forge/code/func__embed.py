import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    base_layer = self.get_base_layer()
    return F.embedding(input, weight, padding_idx=base_layer.padding_idx, max_norm=base_layer.max_norm, norm_type=base_layer.norm_type, scale_grad_by_freq=base_layer.scale_grad_by_freq, sparse=base_layer.sparse)