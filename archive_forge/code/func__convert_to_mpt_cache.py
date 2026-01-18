import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mpt import MptConfig
@staticmethod
def _convert_to_mpt_cache(past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """
        Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
    batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
    batch_size_times_num_heads = batch_size * num_heads
    return tuple(((layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length), layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim)) for layer_past in past_key_value))