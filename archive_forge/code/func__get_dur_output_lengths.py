import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def _get_dur_output_lengths(self, input_ids, dur_out):
    """
        Computes the output length after the duration layer.
        """
    unit_lengths = (input_ids != self.pad_token_id).sum(1)
    unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
    cumulative_dur_out = torch.cumsum(dur_out, dim=1)
    unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()
    return unit_lengths