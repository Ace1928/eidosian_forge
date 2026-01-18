from __future__ import annotations
import math
import warnings
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def _mixed_batch_forward(self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)
    unique_adapters = set(adapter_names)
    sub_batch_indices_list = []
    for adapter in unique_adapters:
        sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
    for i, active_adapter in enumerate(unique_adapters):
        if active_adapter == '__base__':
            continue
        if active_adapter not in self.lora_embedding_A.keys():
            continue
        embedding_A = self.lora_embedding_A[active_adapter].T
        embedding_B = self.lora_embedding_B[active_adapter].T
        scaling = self.scaling[active_adapter]
        sub_batch = x[sub_batch_indices_list[i]]
        after_A = self._embed(sub_batch, embedding_A)
        result[sub_batch_indices_list[i]] += after_A @ embedding_B * scaling
    return result