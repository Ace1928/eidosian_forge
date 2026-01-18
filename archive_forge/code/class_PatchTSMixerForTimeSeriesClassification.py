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
class PatchTSMixerForTimeSeriesClassification(PatchTSMixerPreTrainedModel):
    """
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerLinearHead(config=config)
        self.use_return_dict = config.use_return_dict
        if config.scaling in ['std', 'mean', True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForTimeSeriesClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor=None, output_hidden_states: Optional[bool]=False, return_loss: bool=True, return_dict: Optional[bool]=None) -> PatchTSMixerForTimeSeriesClassificationOutput:
        """
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        loss = torch.nn.CrossEntropyLoss()
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        model_output = self.model(past_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)
        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(model_output.last_hidden_state, loc=model_output.loc, scale=model_output.scale)
        y_hat = self.head(model_output.last_hidden_state)
        if target_values is not None and return_loss is True:
            loss_val = loss(y_hat, target_values)
        else:
            loss_val = None
        if not return_dict:
            return tuple((v for v in [loss_val, y_hat, model_output.last_hidden_state, model_output.hidden_states]))
        return PatchTSMixerForTimeSeriesClassificationOutput(loss=loss_val, prediction_outputs=y_hat, last_hidden_state=model_output.last_hidden_state, hidden_states=model_output.hidden_states)