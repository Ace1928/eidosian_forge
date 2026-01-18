import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
class CLIPSegDecoder(CLIPSegPreTrainedModel):

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)
        self.conditional_layer = config.conditional_layer
        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)
        if config.use_complex_transposed_convolution:
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)
            self.transposed_convolution = nn.Sequential(nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1), nn.ReLU(), nn.ConvTranspose2d(config.reduce_dim, config.reduce_dim // 2, kernel_size=transposed_kernels[0], stride=transposed_kernels[0]), nn.ReLU(), nn.ConvTranspose2d(config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]))
        else:
            self.transposed_convolution = nn.ConvTranspose2d(config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size)
        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList([nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)])
        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = 'relu'
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    def forward(self, hidden_states: Tuple[torch.Tensor], conditional_embeddings: torch.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=True):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        activations = hidden_states[::-1]
        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)
            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(conditional_embeddings)
                output = output.permute(1, 0, 2)
            layer_outputs = layer(output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions)
            output = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states += (output,)
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        output = output[:, 1:, :].permute(0, 2, 1)
        size = int(math.sqrt(output.shape[2]))
        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)
        logits = self.transposed_convolution(output).squeeze()
        if not return_dict:
            return tuple((v for v in [logits, all_hidden_states, all_attentions] if v is not None))
        return CLIPSegDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_attentions)