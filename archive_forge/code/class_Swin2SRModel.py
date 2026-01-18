import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
@add_start_docstrings('The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.', SWIN2SR_START_DOCSTRING)
class Swin2SRModel(Swin2SRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.num_channels == 3 and config.num_channels_out == 3:
            rgb_mean = (0.4488, 0.4371, 0.404)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.img_range = config.img_range
        self.first_convolution = nn.Conv2d(config.num_channels, config.embed_dim, 3, 1, 1)
        self.embeddings = Swin2SREmbeddings(config)
        self.encoder = Swin2SREncoder(config, grid_size=self.embeddings.patch_embeddings.patches_resolution)
        self.layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def pad_and_normalize(self, pixel_values):
        _, _, height, width = pixel_values.size()
        window_size = self.config.window_size
        modulo_pad_height = (window_size - height % window_size) % window_size
        modulo_pad_width = (window_size - width % window_size) % window_size
        pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), 'reflect')
        self.mean = self.mean.type_as(pixel_values)
        pixel_values = (pixel_values - self.mean) * self.img_range
        return pixel_values

    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))
        _, _, height, width = pixel_values.shape
        pixel_values = self.pad_and_normalize(pixel_values)
        embeddings = self.first_convolution(pixel_values)
        embedding_output, input_dimensions = self.embeddings(embeddings)
        encoder_outputs = self.encoder(embedding_output, input_dimensions, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        sequence_output = self.patch_unembed(sequence_output, (height, width))
        sequence_output = self.conv_after_body(sequence_output) + embeddings
        if not return_dict:
            output = (sequence_output,) + encoder_outputs[1:]
            return output
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)