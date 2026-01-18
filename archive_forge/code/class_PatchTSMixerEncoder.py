import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.use_return_dict = config.use_return_dict
        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = PatchTSMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = PatchTSMixerBlock(config=config)
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=None) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        """
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.

                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        patches = self.patcher(past_values)
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)
        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)
        if not return_dict:
            return tuple((v for v in [last_hidden_state, hidden_states]))
        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)