import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class LayoutLMv3Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(config)
        self.output = LayoutLMv3SelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs