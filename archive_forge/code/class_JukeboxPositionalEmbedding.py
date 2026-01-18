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
class JukeboxPositionalEmbedding(nn.Module):

    def __init__(self, embed_dim, width):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty((embed_dim, width)))

    def forward(self):
        pos_emb = self.pos_emb
        return pos_emb