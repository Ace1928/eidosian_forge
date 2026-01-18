import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
def _get_relative_embeddings(self, relative_embeddings, length):
    pad_length = max(length - (self.window_size + 1), 0)
    if pad_length > 0:
        relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
    slice_start_position = max(self.window_size + 1 - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    return relative_embeddings[:, slice_start_position:slice_end_position]