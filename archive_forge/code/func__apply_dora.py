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
def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
    """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
    lora_weight = lora_B.weight @ lora_A.weight
    magnitude = self.lora_magnitude_vector[active_adapter]
    weight = self.get_base_layer().weight
    weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
    weight_norm = weight_norm.detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)
    result_dora = (mag_norm_scale - 1) * F.linear(x, transpose(weight, self.fan_in_fan_out)) + mag_norm_scale * lora_B(lora_A(x)) * scaling
    return result_dora