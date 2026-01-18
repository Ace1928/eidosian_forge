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
def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
    """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
    batch_size = vectors.shape[0]
    split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)
    if len(vectors.shape) == 4:
        return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
    elif len(vectors.shape) == 3:
        return torch.reshape(vectors, split_dim_shape)
    else:
        raise ValueError(f'Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}')