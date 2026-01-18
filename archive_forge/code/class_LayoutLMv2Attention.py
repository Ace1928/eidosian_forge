import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
class LayoutLMv2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs