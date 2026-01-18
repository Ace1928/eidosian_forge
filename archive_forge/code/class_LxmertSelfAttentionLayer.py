import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
class LxmertSelfAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        output = self.self(input_tensor, input_tensor, attention_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs