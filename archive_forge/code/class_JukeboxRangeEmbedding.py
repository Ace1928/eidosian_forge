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
class JukeboxRangeEmbedding(nn.Module):
    """
    The `JukeboxRangeEmbedding` interpolate the given [pos_start, pos_end] to obtain an equivalent of time positional
    embedding of length `n_ctx`.

    Binning process : For each pos in position tensor, find its bin [start,end) mapped to [0,1,...,bins-1] [start,end)
    -> [0,1) -> [0, bins) -> floor -> [0,...,bins-1] NOTE: Open ended interval on right, so start <= pos < end, not <=
    end
    """

    def __init__(self, n_time, embed_dim, range, out_width, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.embed_dim = embed_dim
        self.emb = nn.Embedding(embed_dim, out_width)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        if not len(pos_start.shape) == 2:
            raise TypeError(f'Expected shape with 2 dims, got {pos_start.shape}')
        if not (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all():
            raise TypeError(f'Range is [{self.pos_min},{self.pos_max}), got {pos_start}')
        pos_start = pos_start.float()
        if pos_end is not None:
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)
            pos_end = pos_end.float()
        n_time = self.n_time
        if n_time != 1:
            interpolation = torch.arange(0, n_time, dtype=torch.float, device=pos_start.device).view(1, n_time) / n_time
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        bins_ = (self.embed_dim * normalised_position).floor().long().detach()
        return self.emb(bins_)