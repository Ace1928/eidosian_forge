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
class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    """
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PatchTSMixerPretrainHead(config=config)
        self.masked_loss = config.masked_loss
        self.use_return_dict = config.use_return_dict
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=False, return_loss: bool=True, return_dict: Optional[bool]=None) -> PatchTSMixerForPreTrainingOutput:
        """
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        if self.masked_loss is True:
            loss = torch.nn.MSELoss(reduction='none')
        else:
            loss = torch.nn.MSELoss(reduction='mean')
        model_output = self.model(past_values, observed_mask=observed_mask, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)
        x_hat = self.head(model_output.last_hidden_state)
        if return_loss is True:
            loss_val = loss(x_hat, model_output.patch_input)
        else:
            loss_val = None
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)
        if not return_dict:
            return tuple((v for v in [loss_val, x_hat, model_output.last_hidden_state, model_output.hidden_states]))
        return PatchTSMixerForPreTrainingOutput(loss=loss_val, prediction_outputs=x_hat, last_hidden_state=model_output.last_hidden_state, hidden_states=model_output.hidden_states)