import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaEmbeddings(nn.Module):
    """
    Mega's basic implementation does not incorporate token type embeddings, so this is a stripped-down version of
    RoBERTa's embeddings which optionally includes token types
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.register_buffer('token_type_ids', torch.zeros(config.max_positions, dtype=torch.long).expand((1, -1)), persistent=False)
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Must provide one of input_ids or inputs_embeds')
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        if self.use_token_types:
            if token_type_ids is None:
                if hasattr(self, 'token_type_ids'):
                    buffered_token_type_ids = self.token_type_ids[:, :input_shape[1]]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1])
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds
        return embeddings