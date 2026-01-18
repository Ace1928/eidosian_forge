import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
    """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
    x = x.permute(0, 2, 1, 3)
    return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))