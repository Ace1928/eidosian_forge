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
def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
    weight = weight + scaling * lora_weight
    weight_norm = torch.linalg.norm(weight, dim=1)
    return weight_norm