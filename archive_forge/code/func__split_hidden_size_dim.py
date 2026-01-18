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
def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
    """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
    new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
    x = x.view(*new_x_shape)
    return x.transpose(2, 1)