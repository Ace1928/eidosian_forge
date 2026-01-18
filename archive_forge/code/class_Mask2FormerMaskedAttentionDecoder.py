import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3
        self.decoder_layers = config.decoder_layers - 1
        self.layers = nn.ModuleList([Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        self.mask_predictor = Mask2FormerMaskPredictor(hidden_size=config.hidden_dim, num_heads=config.num_attention_heads, mask_feature_size=self.mask_feature_size)
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds: torch.Tensor=None, multi_stage_positional_embeddings: torch.Tensor=None, pixel_embeddings: torch.Tensor=None, encoder_hidden_states: torch.Tensor=None, query_position_embeddings: torch.Tensor=None, feature_size_list: List=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                The query embeddings that are passed into the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(height*width, batch_size, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        intermediate = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()
        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)
        predicted_mask, attention_mask = self.mask_predictor(intermediate_hidden_states, pixel_embeddings, feature_size_list[0])
        intermediate_mask_predictions += (predicted_mask,)
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                continue
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, None, None, output_attentions)
            else:
                level_index = idx % self.num_feature_levels
                attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
                layer_outputs = decoder_layer(hidden_states, level_index=level_index, position_embeddings=multi_stage_positional_embeddings, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, output_attentions=output_attentions)
                intermediate_hidden_states = self.layernorm(layer_outputs[0])
                predicted_mask, attention_mask = self.mask_predictor(intermediate_hidden_states, pixel_embeddings, feature_size_list[(idx + 1) % self.num_feature_levels])
                intermediate_mask_predictions += (predicted_mask,)
                intermediate += (intermediate_hidden_states,)
            hidden_states = layer_outputs[0]
            if output_attentions:
                attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [hidden_states, all_hidden_states, attentions, intermediate, intermediate_mask_predictions]
            return tuple((v for v in outputs if v is not None))
        return Mask2FormerMaskedAttentionDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=attentions, intermediate_hidden_states=intermediate, masks_queries_logits=intermediate_mask_predictions)