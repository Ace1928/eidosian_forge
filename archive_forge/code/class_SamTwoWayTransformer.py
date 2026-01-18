import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class SamTwoWayTransformer(nn.Module):

    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(self, point_embeddings: Tensor, image_embeddings: Tensor, image_positional_embeddings: Tensor, attention_similarity: Tensor, target_embedding=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_attentions = ()
        if image_embeddings is None:
            raise ValueError('You have to specify an image_embedding')
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        queries = point_embeddings
        keys = image_embeddings
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding
            queries, keys, attention_outputs = layer(queries=queries, keys=keys, query_point_embedding=point_embeddings, key_point_embedding=image_positional_embeddings, attention_similarity=attention_similarity, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return (queries, keys, all_attentions)