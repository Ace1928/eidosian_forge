import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerTransformerModule(nn.Module):
    """
    The MaskFormer's transformer module.
    """

    def __init__(self, in_features: int, config: MaskFormerConfig):
        super().__init__()
        hidden_size = config.decoder_config.hidden_size
        should_project = in_features != hidden_size
        self.position_embedder = MaskFormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.decoder_config.num_queries, hidden_size)
        self.input_projection = nn.Conv2d(in_features, hidden_size, kernel_size=1) if should_project else None
        self.decoder = DetrDecoder(config=config.decoder_config)

    def forward(self, image_features: Tensor, output_hidden_states: bool=False, output_attentions: bool=False, return_dict: Optional[bool]=None) -> DetrDecoderOutput:
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)
        object_queries = self.position_embedder(image_features)
        batch_size = image_features.shape[0]
        queries_embeddings = self.queries_embedder.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.zeros_like(queries_embeddings, requires_grad=True)
        batch_size, num_channels, height, width = image_features.shape
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        object_queries = object_queries.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        decoder_output: DetrDecoderOutput = self.decoder(inputs_embeds=inputs_embeds, attention_mask=None, encoder_hidden_states=image_features, encoder_attention_mask=None, object_queries=object_queries, query_position_embeddings=queries_embeddings, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return decoder_output