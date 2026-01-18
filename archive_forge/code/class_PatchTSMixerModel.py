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
@add_start_docstrings('The PatchTSMixer Model for time-series forecasting.', PATCHTSMIXER_START_DOCSTRING)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):

    def __init__(self, config: PatchTSMixerConfig, mask_input: bool=False):
        super().__init__(config)
        self.use_return_dict = config.use_return_dict
        self.encoder = PatchTSMixerEncoder(config)
        self.patching = PatchTSMixerPatchify(config)
        if mask_input is True:
            self.masking = PatchTSMixerMasking(config)
        else:
            self.masking = None
        if config.scaling == 'mean':
            self.scaler = PatchTSMixerMeanScaler(config)
        elif config.scaling == 'std' or config.scaling is True:
            self.scaler = PatchTSMixerStdScaler(config)
        else:
            self.scaler = PatchTSMixerNOPScaler(config)
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=None) -> PatchTSMixerModelOutput:
        """
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        mask = None
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)
        patched_x = self.patching(scaled_past_values)
        enc_input = patched_x
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
        encoder_output = self.encoder(enc_input, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(encoder_output, tuple):
            encoder_output = PatchTSMixerEncoderOutput(*encoder_output)
        if not return_dict:
            return tuple((v for v in [encoder_output.last_hidden_state, encoder_output.hidden_states, patched_x, mask, loc, scale]))
        return PatchTSMixerModelOutput(last_hidden_state=encoder_output.last_hidden_state, hidden_states=encoder_output.hidden_states, patch_input=patched_x, mask=mask, loc=loc, scale=scale)