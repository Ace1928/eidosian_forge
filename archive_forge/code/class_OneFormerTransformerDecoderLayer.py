import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTransformerDecoderLayer(nn.Module):

    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.cross_attn = OneFormerTransformerDecoderCrossAttentionLayer(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)
        self.self_attn = OneFormerTransformerDecoderSelfAttentionLayer(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)
        self.ffn = OneFormerTransformerDecoderFFNLayer(d_model=self.embed_dim, dim_feedforward=config.dim_feedforward, dropout=0.0, normalize_before=config.pre_norm, layer_norm_eps=config.layer_norm_eps)

    def forward(self, index: int, output: torch.Tensor, multi_stage_features: List[torch.Tensor], multi_stage_positional_embeddings: List[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, query_embeddings: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        """
        Args:
            index (`int`): index of the layer in the Transformer decoder.
            output (`torch.FloatTensor`): the object queries of shape `(N, batch, hidden_dim)`
            multi_stage_features (`List[torch.Tensor]`): the multi-scale features from the pixel decoder.
            multi_stage_positional_embeddings (`List[torch.Tensor]`):
                positional embeddings for the multi_stage_features
            attention_mask (`torch.FloatTensor`): attention mask for the masked cross attention layer
            query_embeddings (`torch.FloatTensor`, *optional*):
                position embeddings that are added to the queries and keys in the self-attention layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        level_index = index % self.num_feature_levels
        attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
        output, cross_attn_weights = self.cross_attn(output, multi_stage_features[level_index], memory_mask=attention_mask, memory_key_padding_mask=None, pos=multi_stage_positional_embeddings[level_index], query_pos=query_embeddings)
        output, self_attn_weights = self.self_attn(output, output_mask=None, output_key_padding_mask=None, query_pos=query_embeddings)
        output = self.ffn(output)
        outputs = (output,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs