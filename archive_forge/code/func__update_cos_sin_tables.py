import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_esm import EsmConfig
def _update_cos_sin_tables(self, x, seq_dimension=2):
    seq_len = x.shape[seq_dimension]
    if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
        self._seq_len_cached = seq_len
        t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]
    return (self._cos_cached, self._sin_cached)