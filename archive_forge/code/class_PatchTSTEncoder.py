import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.embedder = PatchTSTEmbedding(config)
        self.positional_encoder = PatchTSTPositionalEncoding(config, num_patches)
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for i in range(config.num_hidden_layers)])
        self.post_init()

    def forward(self, patch_input: torch.Tensor, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        patch_input = self.embedder(patch_input)
        hidden_state = self.positional_encoder(patch_input)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)
            layer_outputs = encoder_layer(hidden_state=hidden_state, output_attentions=output_attentions)
            hidden_state = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)