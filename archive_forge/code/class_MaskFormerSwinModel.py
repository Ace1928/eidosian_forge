import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils.backbone_utils import BackboneMixin
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerSwinModel(MaskFormerSwinPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = MaskFormerSwinEmbeddings(config)
        self.encoder = MaskFormerSwinEncoder(config, self.embeddings.patch_grid)
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, pixel_values=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))
        embedding_output, input_dimensions = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, input_dimensions, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        hidden_states_spatial_dimensions = (input_dimensions,) + encoder_outputs.hidden_states_spatial_dimensions
        return MaskFormerSwinModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, hidden_states_spatial_dimensions=hidden_states_spatial_dimensions, attentions=encoder_outputs.attentions)