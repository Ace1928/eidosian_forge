import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import get_activation
from ...configuration_utils import PretrainedConfig
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_distilbert import DistilBertConfig
class TransformerBlock(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(f'config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly')
        self.attention = DISTILBERT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            if type(sa_output) != tuple:
                raise TypeError(f'sa_output must be a tuple but it is {type(sa_output)} type')
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output