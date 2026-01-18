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
class OneFormerTransformerDecoder(nn.Module):
    """
    Transformer decoder
    """

    def __init__(self, in_channels: int, config: OneFormerConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.num_heads = config.num_attention_heads
        self.is_training = config.is_training
        self.use_task_norm = config.use_task_norm
        self.use_auxiliary_loss = config.use_auxiliary_loss
        self.query_transformer = OneFormerTransformerDecoderQueryTransformer(d_model=config.hidden_dim, dropout=config.dropout, nhead=config.num_attention_heads, dim_feedforward=config.dim_feedforward, num_decoder_layers=config.query_dec_layers, normalize_before=config.pre_norm, return_intermediate_dec=False, layer_norm_eps=config.layer_norm_eps)
        self.decoder_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.num_feature_levels = 3
        self.layers = nn.ModuleList([OneFormerTransformerDecoderLayer(config) for _ in range(config.decoder_layers - 1)])
        self.query_input_projection = nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.mask_embed = OneFormerMLPPredictionHead(config.hidden_dim, config.hidden_dim, config.mask_dim, 3)

    def forward(self, task_token=None, multi_stage_features=None, multi_stage_positional_embeddings=None, mask_features=None, query_features=None, query_embeddings=None, query_embedder=None, size_list=None, output_attentions=None):
        if self.use_task_norm:
            task_token = self.decoder_norm(task_token)
        object_queries = self.query_transformer(query_features, None, query_embedder.weight[:-1], self.query_input_projection(mask_features), task_token if self.use_task_norm else None)
        object_queries = object_queries[0].permute(1, 0, 2)
        queries = torch.cat([object_queries, task_token], dim=0)
        output = queries.clone()
        intermediate_class_predictions = []
        intermediate_mask_predictions = []
        outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(output, mask_features, attention_mask_target_size=size_list[0])
        intermediate_class_predictions.append(outputs_class)
        intermediate_mask_predictions.append(outputs_mask)
        attentions = ()
        for index, layer in enumerate(self.layers):
            layer_outputs = layer(index=index, output=output, multi_stage_features=multi_stage_features, multi_stage_positional_embeddings=multi_stage_positional_embeddings, attention_mask=attention_mask, query_embeddings=query_embeddings, output_attentions=output_attentions)
            output = layer_outputs[0]
            attentions += (layer_outputs[1:],)
            outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(output, mask_features, attention_mask_target_size=size_list[(index + 1) % self.num_feature_levels])
            intermediate_class_predictions.append(outputs_class)
            intermediate_mask_predictions.append(outputs_mask)
        if not len(intermediate_mask_predictions) == len(self.layers) + 1:
            raise ValueError('Intermediate predictions in the transformer decoder must have the same number of elements as number of layers')
        object_queries = layer_outputs[0].permute(1, 0, 2)
        contrastive_logits = queries.permute(1, 0, 2)
        return OneFormerTransformerDecoderOutput(object_queries=object_queries, contrastive_logits=contrastive_logits, prediction_masks=intermediate_mask_predictions[-1], prediction_class=intermediate_class_predictions[-1], auxiliary_predictions=self._get_aux_predictions(intermediate_class_predictions, intermediate_mask_predictions) if self.use_auxiliary_loss else None, attentions=attentions)

    def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode='bilinear', align_corners=False)
        attention_mask = (attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()
        return (outputs_class, outputs_mask, attention_mask)

    @torch.jit.unused
    def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
        aux_list = [{'class_queries_logits': a, 'masks_queries_logits': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        return tuple(aux_list)