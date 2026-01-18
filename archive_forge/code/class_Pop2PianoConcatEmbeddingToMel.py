import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationConfig
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pop2piano import Pop2PianoConfig
class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """Embedding Matrix for `composer` tokens."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.composer_vocab_size, embedding_dim=config.d_model)

    def forward(self, feature, index_value, embedding_offset):
        index_shifted = index_value - embedding_offset
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds