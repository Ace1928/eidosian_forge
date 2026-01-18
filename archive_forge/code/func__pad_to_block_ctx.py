import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def _pad_to_block_ctx(self, hidden_states, query=False):
    seq_len = hidden_states.shape[1]
    offset = self._offset(seq_len) if query else 0
    n_blocks = (seq_len + offset + self.block_ctx - 1) // self.block_ctx
    pad = n_blocks * self.block_ctx - seq_len - offset
    if pad == 0 and offset == 0:
        return hidden_states
    else:
        return F.pad(hidden_states, (0, 0, offset, pad))